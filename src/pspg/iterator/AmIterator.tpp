#ifndef PSPG_AM_ITERATOR_TPP
#define PSPG_AM_ITERATOR_TPP
// #define double float
/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "AmIterator.h"
#include <pspg/System.h>
#include <util/format/Dbl.h>
#include <pspg/GpuResources.h>
#include <util/containers/FArray.h>
#include <util/misc/Timer.h>
#include <ctime>

//#include <Windows.h>
static
__global__ 
void searchMax(const double* in, double* max, int n)
{
    __shared__ double cache[32];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = -1.0;  // register for each thread
    while (i < n) {
    	if(std::abs(in[i]) > temp)
    		temp = abs(in[i]);
        i += blockDim.x * gridDim.x;  
    }
    cache[cacheIndex] = temp;   // set the cache value 

    __syncthreads();

    // perform parallel reduction, threadsPerBlock must be 2^m

    int ib = blockDim.x / 2;
    while (ib != 0) {
      if(cacheIndex < ib && cache[cacheIndex + ib] > cache[cacheIndex])
        cache[cacheIndex] = cache[cacheIndex + ib]; 

      __syncthreads();

      ib /= 2;
    }
    
    if(cacheIndex == 0)
      max[blockIdx.x] = cache[0];
}

namespace Pscf {
    namespace Pspg {

        using namespace Util;

        template <int D>
        AmIterator<D>::AmIterator()
                : Iterator<D>(),
                  epsilon_(0),
                  lambda_(0),
                  nHist_(0),
                  maxHist_(0),
                  isFlexible_(0)
        {  setClassName("AmIterator"); }

        template <int D>
        AmIterator<D>::AmIterator(System<D>* system)
                : Iterator<D>(system),
                  epsilon_(0),
                  lambda_(0),
                  nHist_(0),
                  maxHist_(0),
                  isFlexible_(0)
        { setClassName("AmIterator"); }

        template <int D>
        AmIterator<D>::~AmIterator()
        {
            delete[] temp_;
            cudaFree(d_temp_);
        }

        template <int D>
        void AmIterator<D>::readParameters(std::istream& in)
        {
            isFlexible_ = 0; //default value (fixed cell)
            read(in, "maxItr", maxItr_);
            read(in, "epsilon", epsilon_);
            read(in, "maxHist", maxHist_);
            readOptional(in, "isFlexible", isFlexible_);
        }

        template <int D>
        void AmIterator<D>::allocate()
        {
            int size_ph = systemPtr_->mesh().size();
            
            devHists_.allocate(maxHist_ + 1);
            omHists_.allocate(maxHist_ + 1);

            if (isFlexible_) {
                devCpHists_.allocate(maxHist_+1);
                CpHists_.allocate(maxHist_+1);
            }

            wArrays_.allocate(systemPtr_->mixture().nMonomer());
            dArrays_.allocate(systemPtr_->mixture().nMonomer());
            tempDev.allocate(systemPtr_->mixture().nMonomer());

            for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i) {
                wArrays_[i].allocate(size_ph);
                dArrays_[i].allocate(size_ph);
                tempDev[i].allocate(size_ph);
            }

            histMat_.allocate(maxHist_ + 1);
            //allocate d_temp_ here i suppose
            cudaMalloc((void**)&d_temp_, NUMBER_OF_BLOCKS * sizeof(cudaReal));
            temp_ = new cudaReal[NUMBER_OF_BLOCKS];
        }

        template <int D>
        int AmIterator<D>::solve()
        {

            // Define Timer objects
            Timer solverTimer;
            Timer stressTimer;
            Timer updateTimer;
            Timer::TimePoint now;

            // Solve MDE for initial state
            solverTimer.start();
            systemPtr_->mixture().compute(systemPtr_->wFieldsRGridPh(),
                                          systemPtr_->cFieldsRGrid());
            // for(int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
            // {
            //     systemPtr_->mixture().polymer(0).block(0).fct().forwardTransform(systemPtr_->wFieldRGridPh(i).cDField());
            // }
            // cudaReal *c_c0, *c_c1;
            // c_c0 = new cudaReal [6];
            // c_c1 = new cudaReal [6];
            // cudaMemcpy(c_c0, systemPtr_->wFieldRGridPh(0).cDField(), 6*sizeof(cudaReal), cudaMemcpyDeviceToHost);
            // cudaMemcpy(c_c1, systemPtr_->wFieldRGridPh(1).cDField(), 6*sizeof(cudaReal), cudaMemcpyDeviceToHost);
            
            // for(int i = 0; i < 6; ++i)
            //     std::cout << c_c0[i]<< "\n";
            // exit(1);

            now = Timer::now();
            solverTimer.stop(now);

            // Compute stress for initial state
            if (isFlexible_) {
                stressTimer.start(now);
                systemPtr_->mixture().computeStress(systemPtr_->wavelist());
                for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m){
                    Log::file() << "Stress    " << m << " = " 
                                << std::setw(21) << std::setprecision(14)
                                << systemPtr_->mixture().stress(m)<<"\n";
                }
                for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m){
                    Log::file() << "Parameter " << m << " = " 
                                << std::setw(21) << std::setprecision(14)
                                << (systemPtr_->unitCell()).parameter(m)<<"\n";
                }
                now = Timer::now();
                stressTimer.stop(now);
            }

            // Anderson-Mixing iterative loop
            int itr;
            for (itr = 1; itr <= maxItr_; ++itr) {
                updateTimer.start(now);
                clock_t time1 = clock();
                if(itr%10 == 0 || itr == 1)
                {
                    Log::file() << "-------------------------------------" << "\n";
                    Log::file() << "iteration #" << itr << "\n";
                }
                if (itr <= maxHist_) {
                    lambda_ = 1.0 - pow(0.95, itr);
                    nHist_ = itr - 1;
                } else {
                    lambda_ = 1.0;
                    nHist_ = maxHist_;
                }

                computeDeviation();

                if (isConverged(itr)) {
                    updateTimer.stop();

                    if (itr > maxHist_ + 1) {
                        invertMatrix_.deallocate();
                        coeffs_.deallocate();
                        vM_.deallocate();
                    }

                    Log::file() << "------- CONVERGED ---------"<< std::endl;

                    // Output final timing results
                    double updateTime = updateTimer.time();
                    double solverTime = solverTimer.time();
                    double stressTime = 0.0;
                    double totalTime = updateTime + solverTime;
                    if (isFlexible_) {
                        stressTime = stressTimer.time();
                        totalTime += stressTime;
                    }
                    Log::file() << "\n";
                    Log::file() << " * Error     = " << Dbl(error_) << '\n';
                    Log::file() << "\n\n";
                    Log::file() << "Iterator times contributions:\n";
                    Log::file() << "\n";
                    Log::file() << "solver time  = " << solverTime  << " s,  "
                                << solverTime/totalTime << "\n";
                    Log::file() << "stress time  = " << stressTime  << " s,  "
                                << stressTime/totalTime << "\n";
                    Log::file() << "update time  = "  << updateTime  << " s,  "
                                << updateTime/totalTime << "\n";
                    Log::file() << "total time   = "  << totalTime   << " s  ";
                    Log::file() << "\n\n";

                    if (isFlexible_) {
                        Log::file() << "\n";
                        Log::file() << "Final stress values:" << "\n";
                        for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m){
                            Log::file() << "Stress    " << m << " = "
                                        << systemPtr_->mixture().stress(m)<<"\n";
                        }
                        Log::file() << "\n";
                        Log::file() << "Final unit cell parameter values:" << "\n";
                        for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m){
                            Log::file() << "Parameter " << m << " = " << std::setprecision(11)
                                        << (systemPtr_->unitCell()).parameter(m)<<"\n";
                        }
                        Log::file() << "\n";
                    }

                    return 0;

                } else {

                    // Resize history based matrix appropriately
                    // consider making these working space local
                    if (itr <= maxHist_ + 1) {
                        if (nHist_ > 0) {
                            invertMatrix_.allocate(nHist_, nHist_);
                            coeffs_.allocate(nHist_);
                            vM_.allocate(nHist_);
                        }
                    }

                    int status = minimizeCoeff(itr);


                    if (status == 1) {
                        //abort the calculations and treat as failure (time out)
                        //perform some clean up stuff
                        invertMatrix_.deallocate();
                        coeffs_.deallocate();
                        vM_.deallocate();
                        return 1;
                    }

                    buildOmega(itr);

                    if (itr <= maxHist_) {
                        if (nHist_ > 0) {
                            invertMatrix_.deallocate();
                            coeffs_.deallocate();
                            vM_.deallocate();
                        }
                    }
                    now = Timer::now();
                    updateTimer.stop(now);

                    // Solve MDE
                    solverTimer.start(now);
                    systemPtr_->mixture().compute(systemPtr_->wFieldsRGridPh(),
                                                  systemPtr_->cFieldsRGrid());
                    // double *phiA_c, *phiB_c;
                    // phiA_c = new double [4];
                    // phiB_c = new double [4];
                    // cudaMemcpy(phiA_c, systemPtr_->cFieldRGrid(0).cDField(), sizeof(double)*4, cudaMemcpyDeviceToHost);
                    // cudaMemcpy(phiB_c, systemPtr_->cFieldRGrid(1).cDField(), sizeof(double)*4, cudaMemcpyDeviceToHost);
                    // for(int i = 0; i < 4; ++i)
                    //     std::cout << phiA_c[i] + phiB_c[i] << std::endl;

                    now = Timer::now();
                    solverTimer.stop(now);

                    if (isFlexible_) {
                        stressTimer.start(now);
                        systemPtr_->mixture().computeStress(systemPtr_->wavelist());
                        if(itr%10 == 0 || itr == 1)
                        {
                            for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m){
                                Log::file() << "Stress    " << m << " = "
                                            << std::setw(21) << std::setprecision(14)
                                            << systemPtr_->mixture().stress(m)<<"\n";
                            }
                            for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m){
                                Log::file() << "Parameter " << m << " = "
                                            << std::setw(21) << std::setprecision(14)
                                            << (systemPtr_->unitCell()).parameter(m)<<"\n";
                            }
                        }
                        now = Timer::now();
                        stressTimer.stop(now);
                    }
                }
                clock_t time2 = clock();
                double t1 = ((double)(time2 - time1)) / CLOCKS_PER_SEC ;
                // std::cout << "iteration time: " << t1 << "s" << std::endl;
            }

            if (itr > maxHist_ + 1) {
                invertMatrix_.deallocate();
                coeffs_.deallocate();
                vM_.deallocate();
            }

            // Failure: Not converged after maxItr iterations.
            return 1;
        }
        
        template <int D>
        void AmIterator<D>::computeDeviation()
        {
            int size_ph = systemPtr_->mesh().size();

            if (isFlexible_) {
                CpHists_.append(systemPtr_->unitCell().parameters());
            } 

            // need to average
            double average = 0.0;
            for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i) {
                average += reductionH(systemPtr_->wFieldRGridPh(i), size_ph);
            }
            average /= (systemPtr_->mixture().nMonomer() * size_ph);
            for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i) {
                subtractUniform << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > 
                (systemPtr_->wFieldRGridPh(i).cDField(), average, size_ph);
            }

            omHists_.append(systemPtr_->wFieldsRGridPh());

            for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i) {
                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (tempDev[i].cDField(), 0, size_ph);
            }

            for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i) {
                for (int j = 0; j < systemPtr_->mixture().nMonomer(); ++j) {
                    pointWiseAddScale <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (tempDev[i].cDField(),
                     systemPtr_->cFieldRGrid(j).cDField(),
                     systemPtr_->interaction().chi(i, j),
                     size_ph);

                    pointWiseAddScale <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (tempDev[i].cDField(),
                     systemPtr_->wFieldRGridPh(j).cDField(),
                     -systemPtr_->interaction().idemp(i, j),
                     size_ph);
                }
            }
            
            // double sum_chi_inv = (double) systemPtr_->interaction().sum_inv();
            // for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i) 
            // {
            //     // std::cout << systemPtr_->interaction().sum_inv() << "\n";
            //     pointWiseSubtractFloat <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            //     (tempDev[i].cDField(),
            //      1.0/sum_chi_inv, 
            //      size_ph);
            // }
            // Average
            double average = 0.0;
            for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i) 
            {
                average += reductionH(tempDev[i], size_ph);
            }
            
            average /= (systemPtr_->mixture().nMonomer() * size_ph);
            
            for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i) 
            {
                subtractUniform<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (tempDev[i].cDField(), average, size_ph);
            }

            devHists_.append(tempDev);
            
            if (isFlexible_)
            {
                FArray<double, 6> tempCp;
                for (int i = 0; i<(systemPtr_->unitCell()).nParameter(); i++)
                {
                    //format????
                    tempCp [i] = -((systemPtr_->mixture()).stress(i));
                }
                devCpHists_.append(tempCp);
            }
        }

        template <int D>
        bool AmIterator<D>::isConverged(int itr)
        {
            double error;
            double dError = 0;
            double wError = 0;

            int size_ph = systemPtr_->mesh().size();
            
            // for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i) {
            //     dError += innerProduct(devHists_[0][i], devHists_[0][i], size_ph);
            //     wError += innerProduct(systemPtr_->wFieldRGridPh(i),
            //                            systemPtr_->wFieldRGridPh(i),
            //                            size_ph);
            // }

            // if (isFlexible_) {
            //     for ( int i = 0; i < systemPtr_->unitCell().nParameter(); i++) {
            //         dError +=  devCpHists_[0][i] *  devCpHists_[0][i];
            //         wError +=  systemPtr_->unitCell().parameter(i) * systemPtr_->unitCell().parameter(i);
            //     }
            // }
            // error = sqrt(dError / wError);
            // error_ = error;

            double maxErr = 0.0;
            double *max_g, *max_c;
            cudaMalloc((void**)&max_g, NUMBER_OF_BLOCKS * sizeof(double));
            max_c = new double [NUMBER_OF_BLOCKS];
            for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i) 
            {
                searchMax<<<32, 32>>>(devHists_[0][i].cDField(), max_g, size_ph);  
                cudaMemcpy(max_c, max_g, sizeof(double)*NUMBER_OF_BLOCKS, cudaMemcpyDeviceToHost);  
                for (int j = 0; j < NUMBER_OF_BLOCKS; ++j)
                {
                    if (max_c[j] > maxErr)
                        maxErr = max_c[j];
                }
            }

            if (isFlexible_) {
                for ( int i = 0; i < systemPtr_->unitCell().nParameter(); i++) {
                    if (abs(devCpHists_[0][i]) > maxErr)
                        maxErr = abs(devCpHists_[0][i]);
                }
            }
            
            error = maxErr;
            error_ = error;
            cudaFree(max_g);
            free(max_c);
            if(itr%10 == 0 || itr == 1)
            {
                // Log::file() << " dError :" << Dbl(dError) << '\n';
                // Log::file() << " wError :" << Dbl(wError) << '\n';
                Log::file() << "  Error :" << Dbl(error) << '\n';
            }
        
            if (error < epsilon_) {
                final_error = error;
                return true;
            }
            else {
                return false;
            }
            
        }

        template <int D>
        int AmIterator<D>::minimizeCoeff(int itr)
        {   
            int size_ph = systemPtr_->mesh().size();;

            if (itr == 1) {
                //do nothing
                histMat_.reset();
                return 0;
            }
            else {

                int nMonomer = systemPtr_->mixture().nMonomer();
                int nParameter = systemPtr_->unitCell().nParameter();

                double elm, elm_cp;
                //clear last column and shift everything downwards if necessary
                histMat_.clearColumn(nHist_);
                //calculate the new values for d(k)d(k') matrix

                for (int i = 0; i < nHist_; ++i){
                    for (int j = i; j < nHist_; ++j){
                        invertMatrix_(i,j) = 0;
                        for (int k = 0; k < nMonomer; ++k){
                            elm = 0;
                            RDField<D> temp1, temp2;
                            temp1.allocate(size_ph);
                            temp2.allocate(size_ph);
                            pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                            (devHists_[0][k].cDField(), devHists_[i+1][k].cDField(),
                             temp1.cDField(), size_ph);
                            pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                            (devHists_[0][k].cDField(), devHists_[j+1][k].cDField(),
                             temp2.cDField(), size_ph);
                            elm += AmIterator<D>::innerProduct(temp1, temp2,
                                                               size_ph);
                            
                            temp1.deallocate();
                            temp2.deallocate();
                            
                            invertMatrix_(i,j) += elm;
                        }

                        if (isFlexible_){
                            elm_cp = 0;
                            for (int m = 0; m < nParameter; ++m){
                                elm_cp += ((devCpHists_[0][m] - devCpHists_[i+1][m]) *
                                           (devCpHists_[0][m] - devCpHists_[j+1][m]));
                            }
                            invertMatrix_(i, j) += elm_cp;
                        }
                        invertMatrix_(j,i) = invertMatrix_(i,j);
                    }

                    vM_[i] = 0;
                    for (int j = 0; j < nMonomer; ++j) {
                        RDField<D> temp;
                        temp.allocate(size_ph);
                        pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                        (devHists_[0][j].cDField(), devHists_[i + 1][j].cDField(),
                         temp.cDField(), size_ph);
                        vM_[i] += AmIterator<D>::innerProduct(temp, devHists_[0][j],
                                                                  size_ph);
                        temp.deallocate();
                    }
                    if (isFlexible_)
                    {
                        elm_cp = 0;
                        for (int m = 0; m < nParameter ; ++m)
                        {
                            vM_[i] += ((devCpHists_[0][m] - devCpHists_[i + 1][m]) *
                                       (devCpHists_[0][m]));
                        }
                    }
                }

                if (itr == 2) {
                    coeffs_[0] = vM_[0] / invertMatrix_(0, 0);
                    // std::cout << vM_[0] << "\n";
                    // std::cout << invertMatrix_(0, 0) << "\n";
                    // std::cout << coeffs_[0]  << "\n";
                }
                else {

                    LuSolver solver;
                    solver.allocate(nHist_);
                    solver.computeLU(invertMatrix_);

                    /*
                    int status = solver.solve(vM_, coeffs_);
                    if (status) {
                       if (status == 1) {
                          //matrix is singular do something
                          return 1;
                       }
                       }*/
                    solver.solve(vM_, coeffs_);
                    //for the sake of simplicity during porting
                    //we leaves out checks for singular matrix here
                    //--GK 09 11 2019

                }
                return 0;
            }
        }

        template <int D>
        cudaReal AmIterator<D>::innerProduct(const RDField<D>& a, const RDField<D>& b, int size) {

            switch(THREADS_PER_BLOCK){
                case 512:
                    deviceInnerProduct<512>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                    (d_temp_, a.cDField(), b.cDField(), size);
                    break;
                case 256:
                    deviceInnerProduct<256>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                    (d_temp_, a.cDField(), b.cDField(), size);
                    break;
                case 128:
                    deviceInnerProduct<128>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                    (d_temp_, a.cDField(), b.cDField(), size);
                    break;
                case 64:
                    deviceInnerProduct<64>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                    (d_temp_, a.cDField(), b.cDField(), size);
                    break;
                case 32:
                    deviceInnerProduct<32>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                    (d_temp_, a.cDField(), b.cDField(), size);
                    break;
                case 16:
                    deviceInnerProduct<16>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                    (d_temp_, a.cDField(), b.cDField(), size);
                    break;
                case 8:
                    deviceInnerProduct<8>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                    (d_temp_, a.cDField(), b.cDField(), size);
                    break;
                case 4:
                    deviceInnerProduct<4>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                    (d_temp_, a.cDField(), b.cDField(), size);
                    break;
                case 2:
                    deviceInnerProduct<2>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                    (d_temp_, a.cDField(), b.cDField(), size);
                    break;
                case 1:
                    deviceInnerProduct<1>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                    (d_temp_, a.cDField(), b.cDField(), size);
                    break;
            }
            cudaMemcpy(temp_, d_temp_, NUMBER_OF_BLOCKS * sizeof(cudaReal), cudaMemcpyDeviceToHost);
            cudaReal final = 0;
            // cudaReal c = 0;
            // // use kahan summation to reduce error
            //  for (int i = 0; i < NUMBER_OF_BLOCKS; ++i) {
            //     cudaReal y = temp_[i] - c;
            //     cudaReal t = final + y;
            //     c = (t - final) - y;
            //     final = t;
            //  }
           for(int i = 0; i < NUMBER_OF_BLOCKS; ++i)
           {
               final += temp_[i];
           }

            return final;
        }

        template <int D>
        cudaReal AmIterator<D>::innerProduct2(const RDField<D>& a, const RDField<D>& b, int size) {

            switch(THREADS_PER_BLOCK){
                case 512:
                    deviceInnerProduct<512>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_temp_, a.cDField()+1, b.cDField()+1, size);
                    break;
                case 256:
                    deviceInnerProduct<256>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_temp_, a.cDField()+1, b.cDField()+1, size);
                    break;
                case 128:
                    deviceInnerProduct<128>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_temp_, a.cDField()+1, b.cDField()+1, size);
                    break;
                case 64:
                    deviceInnerProduct<64>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_temp_, a.cDField()+1, b.cDField()+1, size);
                    break;
                case 32:
                    deviceInnerProduct<32>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_temp_, a.cDField()+1, b.cDField()+1, size);
                    break;
                case 16:
                    deviceInnerProduct<16>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_temp_, a.cDField(), b.cDField(), size);
                    break;
                case 8:
                    deviceInnerProduct<8>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_temp_, a.cDField()+1, b.cDField()+1, size);
                    break;
                case 4:
                    deviceInnerProduct<4>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_temp_, a.cDField()+1, b.cDField(), size);
                    break;
                case 2:
                    deviceInnerProduct<2>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_temp_, a.cDField(), b.cDField()+1, size);
                    break;
                case 1:
                    deviceInnerProduct<1>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_temp_, a.cDField()+1, b.cDField()+1, size);
                    break;
            }
            cudaMemcpy(temp_, d_temp_, NUMBER_OF_BLOCKS * sizeof(cudaReal), cudaMemcpyDeviceToHost);
            cudaReal final = 0;
            /*cudaReal c = 0;*/
            //use kahan summation to reduce error
            // for (int i = 0; i < NUMBER_OF_BLOCKS; ++i) {
            //    cudaReal y = temp_[i] - c;
            //    cudaReal t = final + y;
            //    c = (t - final) - y;
            //    final = t;
            // }
            for(int i = 0; i < NUMBER_OF_BLOCKS; ++i)
            {
                final += temp_[i];
            }

            return final;
        }

        template<int D>
        cudaReal AmIterator<D>::reductionH(RDField<D>& a, int size)
        {
            reduction <<< NUMBER_OF_BLOCKS , THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(cudaReal) >>>
            (d_temp_, a.cDField(), size);
            cudaMemcpy(temp_, d_temp_, NUMBER_OF_BLOCKS  * sizeof(cudaReal), cudaMemcpyDeviceToHost);
            cudaReal final = 0;
            /*cudaReal c = 0;*/
            // for (int i = 0; i < NUMBER_OF_BLOCKS/2 ; ++i) {
            //    cudaReal y = temp_[i] - c;
            //    cudaReal t = final + y;
            //    c = (t - final) - y;
            //    final = t;
            // }
            for (int i = 0; i < NUMBER_OF_BLOCKS ; ++i)
            {
                final += temp_[i];
            }
            return final;
        }

        template <int D>
        void AmIterator<D>::buildOmega(int itr)
        {  
            int size_ph = systemPtr_->mesh().size();

            if (itr == 1) {
                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i)
                {
                    assignReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (systemPtr_->wFieldRGridPh(i).cDField(),
                     omHists_[0][i].cDField(), size_ph);
                    pointWiseAddScale <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (systemPtr_->wFieldRGridPh(i).cDField(),
                     devHists_[0][i].cDField(), lambda_, size_ph);

                }

                // for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i) 
                // {
                //     assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                //     (tempDev[i].cDField(), 0, 1);
                // }

                // for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i) 
                // {
                //     for (int j = 0; j < systemPtr_->mixture().nMonomer(); ++j) 
                //     {
                //         if(i != j)
                //         {
                //             pointWiseAddScale <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                //             (tempDev[i].cDField(),
                //             systemPtr_->wFieldRGrid(j).cDField(),
                //             systemPtr_->interaction().chi(i, j),
                //             1);
                //         }
                
                //         assignReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                //         (systemPtr_->wFieldRGrid(i).cDField(), tempDev[i].cDField(), 1);
                //     }
                // }


                if (isFlexible_) {
                    cellParameters_.clear();
                    for (int m = 0; m < (systemPtr_->unitCell()).nParameter() ; ++m){
                        cellParameters_.append(CpHists_[0][m] +lambda_* devCpHists_[0][m]);
                    }
                    systemPtr_->unitCell().setParameters(cellParameters_);
                    systemPtr_->mixture().setupUnitCell(systemPtr_->unitCell(), systemPtr_->wavelist());
                    systemPtr_->wavelist().computedKSq(systemPtr_->unitCell());
                }

            } else {
                //should be strictly correct. coeffs_ is a vector of size 1 if itr ==2

                for (int j = 0; j < systemPtr_->mixture().nMonomer(); ++j) {
                    assignReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (wArrays_[j].cDField(), omHists_[0][j].cDField(),  size_ph);
                    assignReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (dArrays_[j].cDField(), devHists_[0][j].cDField(),  size_ph);
                }

                for (int i = 0; i < nHist_; ++i) {
                    for (int j = 0; j < systemPtr_->mixture().nMonomer(); ++j) {
                        //wArrays
                        pointWiseBinarySubtract <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                                (omHists_[i + 1][j].cDField(),
                                 omHists_[0][j].cDField(),
                                 tempDev[0].cDField(),
                                 size_ph);

                        pointWiseAddScale <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                                (wArrays_[j].cDField(),
                                 tempDev[0].cDField(), coeffs_[i],
                                 size_ph);

                        //dArrays
                        pointWiseBinarySubtract <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                        (devHists_[i + 1][j].cDField(),
                         devHists_[0][j].cDField(), tempDev[0].cDField(),
                         size_ph);
                        pointWiseAddScale <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                        (dArrays_[j].cDField(),
                         tempDev[0].cDField(), coeffs_[i], size_ph);
                    }
                    // std::cout << coeffs_[i] << "\n";
                }

                for (int i = 0; i < systemPtr_->mixture().nMonomer(); ++i) {
                    assignReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (systemPtr_->wFieldRGridPh(i).cDField(),
                     wArrays_[i].cDField(), size_ph);
                    pointWiseAddScale <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (systemPtr_->wFieldRGridPh(i).cDField(),
                     dArrays_[i].cDField(), lambda_, size_ph);
                }

                // cudaReal c_c[32];
                // cudaMemcpy(c_c, systemPtr_->wFieldRGridPh(1).cDField(), 32*sizeof(cudaReal), cudaMemcpyDeviceToHost);
                // for(int i = 0; i < 32; ++i)
                //     std::cout << c_c[i] << "\n";
                // std::cout << "\n";
                // exit(1);

                if (isFlexible_) {

                    for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m){
                        wCpArrays_[m] = CpHists_[0][m];
                        dCpArrays_[m] = devCpHists_[0][m];
                    }
                    for (int i = 0; i < nHist_; ++i) {
                        for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m) {
                            wCpArrays_[m] += coeffs_[i] * ( CpHists_[i+1][m]-
                                                            CpHists_[0][m]);
                            dCpArrays_[m] += coeffs_[i] * ( devCpHists_[i+1][m]-
                                                            devCpHists_[0][m]);
                        }
                    }

                    cellParameters_.clear();
                    for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m){
                        cellParameters_.append(wCpArrays_[m] + lambda_* dCpArrays_[m]);
                    }

                    systemPtr_->unitCell().setParameters(cellParameters_);
                    systemPtr_->mixture().setupUnitCell(systemPtr_->unitCell(), systemPtr_->wavelist());
                    systemPtr_->wavelist().computedKSq(systemPtr_->unitCell());

                }


            }
        }

    }
}
#endif
