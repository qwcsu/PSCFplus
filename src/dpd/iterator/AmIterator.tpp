#ifndef DPD_AM_ITERATOR_TPP
#define DPD_AM_ITERATOR_TPP

#include "AmIterator.h"
#include <dpd/System.h>
#include <pspg/GpuResources.h>
#include <util/format/Dbl.h>
#include <util/containers/FArray.h>
#include <util/misc/Timer.h>
#include <ctime>


__global__
void scaleReal2(cudaReal* a, cudaReal* scale, int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startID; i < size; i += nThreads)
    {
        a[i] *= scale[i];
    }
}

namespace Pscf {
namespace Pspg {
    namespace DPD
    {
        using namespace Util;

        template <int D>
        AmIterator<D>::AmIterator()
        : Iterator<D>(),
          epsilon_(0),
          lambda_(0),
          nHist_(0),
          maxHist_(0),
          isFlexible_(0)
        {
            setClassName("AmIterator");
        }

        template <int D>
        AmIterator<D>::AmIterator(System<D>* system)
        : Iterator<D>(system),
          epsilon_(0),
          lambda_(0),
          nHist_(0),
          maxHist_(0),
          isFlexible_(0)
        {
            setClassName("AmIterator");
        }

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
            devHists_.allocate(maxHist_ + 1);
            omHists_.allocate(maxHist_ + 1);
            histMat_.allocate(maxHist_ + 1);

            if (isFlexible_) 
            {
                devCpHists_.allocate(maxHist_+1);
                CpHists_.allocate(maxHist_+1);
            }

            wArrays_.allocate(systemPtr_->dpdmixture().nMonomer());
            dArrays_.allocate(systemPtr_->dpdmixture().nMonomer());
            tempDev.allocate(systemPtr_->dpdmixture().nMonomer());

            for (int i = 0; i < systemPtr_->dpdmixture().nMonomer(); ++i) 
            {
                wArrays_[i].allocate(systemPtr_->basis().nStar());
                dArrays_[i].allocate(systemPtr_->basis().nStar());
                tempDev[i].allocate(systemPtr_->basis().nStar());
            }

            cudaMalloc((void**)&d_temp_, NUMBER_OF_BLOCKS * sizeof(cudaReal));
            temp_ = new cudaReal[NUMBER_OF_BLOCKS];
        }

        template <int D>
        int AmIterator<D>::solve()
        {
            
            systemPtr_->dpdmixture().compute(systemPtr_->wFieldsRGrid(), 
                                             systemPtr_->cFieldsRGrid());
            
            systemPtr_->fieldIo().convertRGridToBasis(systemPtr_->cFieldsRGrid(),
                                                      systemPtr_->cFields());
            systemPtr_->fieldIo().convertRGridToBasis(systemPtr_->wFieldsRGrid(),
                                                      systemPtr_->wFields());
        
            // int ns = systemPtr_->basis().nStar();
            // double w[ns];
            // double cA[ns];
            // double cB[ns];
            // double u[ns];
            // cudaMemcpy(w,
            //            systemPtr_->wField(0).cDField(),
            //            ns * sizeof(cudaReal),
            //            cudaMemcpyDeviceToHost);
            // cudaMemcpy(cA,
            //            systemPtr_->cField(0).cDField(),
            //            ns * sizeof(cudaReal),
            //            cudaMemcpyDeviceToHost);
            // cudaMemcpy(cB,
            //            systemPtr_->cField(1).cDField(),
            //            ns * sizeof(cudaReal),
            //            cudaMemcpyDeviceToHost);
            // cudaMemcpy(u,
            //            systemPtr_->dpddiblock().bu0().cDField(),
            //            ns * sizeof(cudaReal),
            //            cudaMemcpyDeviceToHost);
            // for(int j = 0; j < ns; ++j)
            // {
            //     std::cout << "w[" << j << "] = " << w[j] << "    cA[" << j << "] = " << cA[j] <<"  " << cB[j] << "  " << u[j]<< "\n";
            // }
            // exit(1);                             
            if(isFlexible_)
            {   
                // exit(1);
                systemPtr_->dpdmixture().computeStress(systemPtr_->wavelist());
                for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m){
                    Log::file() << "Stress    " << m << " = "
                                 << std::setw(21) << std::setprecision(14)
                                << systemPtr_->dpdmixture().stress(m)<<"\n";
                }
                for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m){
                    Log::file() << "Parameter " << m << " = "
                                << std::setw(21) << std::setprecision(14)
                                << (systemPtr_->unitCell()).parameter(m)<<"\n";
                }
            }

            int itr;
            for(itr = 1; itr <= maxItr_; ++itr)
            {
                if(itr%50 == 0 || itr == 1)
                {
                    Log::file() << "-------------------------------------" << "\n";
                    Log::file() << "iteration #" << itr << "\n";
                }
                if (itr <= maxHist_) 
                {
                    lambda_ = 1.0 - pow(0.9, itr);
                    nHist_ = itr - 1;
                } 
                else 
                {
                    lambda_ = 1.0 ;//- pow(0.1, maxHist_);
                    nHist_ = maxHist_;
                }

                computeDeviation();

                if (isConverged(itr)) 
                {
                    if (itr > maxHist_ + 1) 
                    {
                        invertMatrix_.deallocate();
                        coeffs_.deallocate();
                        vM_.deallocate();
                    }

                    if (isFlexible_) {
                        Log::file() << "\n";
                        Log::file() << "Final stress values:" << "\n";
                        for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m){
                            Log::file() << "Stress    " << m << " = "
                                        << std::setw(21) << std::setprecision(14)
                                        << systemPtr_->dpdmixture().stress(m)<<"\n";
                        }
                        Log::file() << "\n";
                        Log::file() << "Final unit cell parameter values:" << "\n";
                        for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m){
                            Log::file() << "Parameter " << m << " = "
                                        << std::setw(21) << std::setprecision(14)
                                        << (systemPtr_->unitCell()).parameter(m)<<"\n";
                        }
                        Log::file() << "\n";
                    }


                    return 0;
                }
                else
                {
                    if (itr <= maxHist_ + 1)
                    {
                        if (nHist_ > 0)
                        {
                            invertMatrix_.allocate(nHist_, nHist_);
                            coeffs_.allocate(nHist_);
                            vM_.allocate(nHist_);
                        }
                    }

                    int status = minimizeCoeff(itr);

                    if (status == 1) 
                    {
                        invertMatrix_.deallocate();
                        coeffs_.deallocate();
                        vM_.deallocate();
                        return 1;
                    }

                    buildOmega(itr);
// exit(1);
                    if (itr <= maxHist_) 
                    {
                        if (nHist_ > 0) 
                        {
                            invertMatrix_.deallocate();
                            coeffs_.deallocate();
                            vM_.deallocate();
                        }
                    }

                    systemPtr_->fieldIo().convertBasisToRGrid(systemPtr_->wFields(),
                                                              systemPtr_->wFieldsRGrid());

                    systemPtr_->dpdmixture().compute(systemPtr_->wFieldsRGrid(),
                                                    systemPtr_->cFieldsRGrid());
                    
                    systemPtr_->fieldIo().convertRGridToBasis(systemPtr_->cFieldsRGrid(),
                                                              systemPtr_->cFields());

                    if (isFlexible_) 
                    {
                        systemPtr_->dpdmixture().computeStress(systemPtr_->wavelist());
                        if(itr%50 == 0 || itr == 1)
                        {
                            for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m){
                                Log::file() << "Stress    " << m << " = "
                                            << std::setw(21) << std::setprecision(14)
                                            << systemPtr_->dpdmixture().stress(m)<<"\n";
                            }
                            for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m){
                                Log::file() << "Parameter " << m << " = "
                                            << std::setw(21) << std::setprecision(14)
                                            << (systemPtr_->unitCell()).parameter(m)<<"\n";
                            }
                        }
                    }

                    // double c[systemPtr_->mesh().size()];
                    // cudaMemcpy(c, systemPtr_->cFieldRGrid(0).cDField(),
                    //           sizeof(cudaReal)*systemPtr_->mesh().size(),
                    //           cudaMemcpyDeviceToHost);
                    // for(int i = 0; i < systemPtr_->mesh().size(); ++i)
                    //     std::cout << c[i] << std::endl;
                
                    systemPtr_->fieldIo().convertRGridToBasis(systemPtr_->cFieldsRGrid(),
                                                              systemPtr_->cFields());

                }
            }
            if (itr > maxHist_ + 1) {
                invertMatrix_.deallocate();
                coeffs_.deallocate();
                vM_.deallocate();
            }

            return 1;
        }

        template <int D>
        void AmIterator<D>::computeDeviation()
        {
            
        }

        template <int D>
        bool AmIterator<D>::isConverged(int itr)
        {
            double error;
            double dError = 0;
            // double wError = 0;
            // double max_error = 0;
            int nStar = systemPtr_->basis().nStar();

            for (int i = 0; i < systemPtr_->dpdmixture().nMonomer(); ++i) 
            {
                // dError += innerProduct(devHists_[0][i], devHists_[0][i], systemPtr_->basis().nStar());
                // wError += innerProduct(systemPtr_->wField(i),
                //                        systemPtr_->wField(i),
                //                        systemPtr_->basis().nStar());
                double hostHists[nStar];
                cudaMemcpy(hostHists, devHists_[0][i].cDField(), sizeof(double)*nStar, cudaMemcpyDeviceToHost);
                for (int j = 0; j < systemPtr_->basis().nStar(); ++j)
                {
                    double absHostHists = abs(hostHists[j]);
                    if (absHostHists > dError)
                        dError = absHostHists;
                }        
            }
            // std::cout << dError << std::endl;
            // exit(1);

            if (isFlexible_) 
            {
                for ( int i = 0; i < systemPtr_->unitCell().nParameter(); i++) 
                {
                    // dError += devCpHists_[0][i] *  devCpHists_[0][i];
                    // // wError += systemPtr_->unitCell().nParameter();
                    // wError +=  systemPtr_->unitCell().parameter(i) * systemPtr_->unitCell().parameter(i);
                    double absHostCpHists = abs(devCpHists_[0][i]);
                    if (absHostCpHists > dError)
                        dError = absHostCpHists;
                }
            }
            // std::cout << "max_error" << dError << "\n";
            // exit(1);

            // error = sqrt(dError / wError);
            error = dError;
            if(itr%50 == 0 || itr == 1)
            {
                // Log::file() << " dError :" << Dbl(dError) << '\n';
                // Log::file() << " wError :" << Dbl(wError) << '\n';
                Log::file() << "Error :" << Dbl(error) << '\n';
            }

            if (error < epsilon_) 
            {
                Log::file() << std::endl << "------- CONVERGED ---------"<< std::endl;
                Log::file() << "Final error        : " << Dbl(error) << "\n";
                final_error = error;
                return true;
            }
            else 
            {
                return false;
            }
        }

        template <int D>
        int AmIterator<D>::minimizeCoeff(int itr)
        {
            if (itr == 1)
            {
                histMat_.reset();
                return 0;
            }
            else
            {
                int nMonomer = systemPtr_->dpdmixture().nMonomer();
                int nParameter = systemPtr_->unitCell().nParameter();
                int nStar = systemPtr_->basis().nStar();

                double elm, elm_cp;

                histMat_.clearColumn(nHist_);
                //calculate the new values for d(k)d(k') matrix

                for (int i = 0; i < nHist_; ++i){
                    for (int j = i; j < nHist_; ++j){
                        invertMatrix_(i,j) = 0;
                        for (int k = 0; k < nMonomer; ++k){
                            elm = 0;
                            RDField<D> temp1, temp2;
                            temp1.allocate(systemPtr_->basis().nStar());
                            temp2.allocate(systemPtr_->basis().nStar());
                            pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                            (devHists_[0][k].cDField(), devHists_[i+1][k].cDField(),
                             temp1.cDField(), systemPtr_->basis().nStar());
                            pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                            (devHists_[0][k].cDField(), devHists_[j+1][k].cDField(),
                             temp2.cDField(), systemPtr_->basis().nStar());
                            elm += AmIterator<D>::innerProduct(temp1, temp2,
                                                               systemPtr_->basis().nStar());
                            temp1.deallocate();
                            temp2.deallocate();
                            invertMatrix_(i,j) += elm;
                        }
                        
                        if (isFlexible_)
                        {
                            elm_cp = 0;
                            for (int m = 0; m < nParameter; ++m)
                            {
                                elm_cp += ((devCpHists_[0][m] - devCpHists_[i+1][m]) *
                                           (devCpHists_[0][m] - devCpHists_[j+1][m]));
                            }
                            invertMatrix_(i, j) += elm_cp;   
                        }
                        invertMatrix_(j,i) = invertMatrix_(i,j);
                    }
                }

                for (int i = 0; i < nHist_; ++i)
                {
                    vM_[i] = 0;
                    for (int j = 0; j < nMonomer; ++j) 
                    {
                        RDField<D> temp;
                        temp.allocate(systemPtr_->basis().nStar());
                        pointWiseBinarySubtract<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                        (devHists_[0][j].cDField(), devHists_[i + 1][j].cDField(),
                         temp.cDField(), systemPtr_->basis().nStar());
                        vM_[i] += AmIterator<D>::innerProduct(temp, devHists_[0][j],
                                                                  systemPtr_->basis().nStar());
                        temp.deallocate();
                    }
                    // vM_[i] = histMat_.makeVm(i, nHist_);
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

                if (itr == 2) 
                {
                    coeffs_[0] = vM_[0] / invertMatrix_(0, 0);
                }
                else 
                {
                    LuSolver solver;
                    solver.allocate(nHist_);
                    solver.computeLU(invertMatrix_);
                    solver.solve(vM_, coeffs_);
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
            cudaReal c = 0;
            // use kahan summation to reduce error
             for (int i = 0; i < NUMBER_OF_BLOCKS; ++i) {
                cudaReal y = temp_[i] - c;
                cudaReal t = final + y;
                c = (t - final) - y;
                final = t;
             }
        //    for(int i = 0; i < NUMBER_OF_BLOCKS; ++i)
        //    {
        //        final += temp_[i];
        //    }

            return final;
        }


        template <int D>
        void AmIterator<D>::buildOmega(int itr)
        {
            if (itr == 1) 
            {
                for (int i = 0; i < systemPtr_->dpdmixture().nMonomer(); ++i)
                {
                    assignReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (systemPtr_->wField(i).cDField(),
                     omHists_[0][i].cDField(), systemPtr_->basis().nStar());
                    pointWiseAddScale <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (systemPtr_->wField(i).cDField(),
                     devHists_[0][i].cDField(), lambda_, systemPtr_->basis().nStar());
                }

                if (isFlexible_) 
                {
                    cellParameters_.clear();
                    for (int m = 0; m < (systemPtr_->unitCell()).nParameter() ; ++m)
                    {
                        cellParameters_.append(CpHists_[0][m] +lambda_* devCpHists_[0][m]);
                        // std::cout<< "cell param : " << CpHists_[0][m] +lambda_* devCpHists_[0][m] << "\n";
                    }
        
                    systemPtr_->unitCell().setParameters(cellParameters_);
                    systemPtr_->dpdmixture().setupUnitCell(systemPtr_->unitCell(), systemPtr_->wavelist());
                    systemPtr_->wavelist().computedKSq(systemPtr_->unitCell());
                }
            }
            else
            {
                for (int j = 0; j < systemPtr_->dpdmixture().nMonomer(); ++j) 
                {
                    assignReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>(wArrays_[j].cDField(),
                            omHists_[0][j].cDField(),  systemPtr_->basis().nStar());
                    assignReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>(dArrays_[j].cDField(),
                            devHists_[0][j].cDField(),  systemPtr_->basis().nStar());
                }
                for (int i = 0; i < nHist_; ++i) 
                {
                    for (int j = 0; j < systemPtr_->dpdmixture().nMonomer(); ++j) 
                    {
                        pointWiseBinarySubtract <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                                (omHists_[i + 1][j].cDField(),
                                 omHists_[0][j].cDField(),
                                 tempDev[0].cDField(),
                                 systemPtr_->basis().nStar());
                        pointWiseAddScale <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                                (wArrays_[j].cDField(),
                                 tempDev[0].cDField(), coeffs_[i],
                                 systemPtr_->basis().nStar());
                        pointWiseBinarySubtract <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                        (devHists_[i + 1][j].cDField(),
                         devHists_[0][j].cDField(), tempDev[0].cDField(),
                         systemPtr_->basis().nStar());
                        pointWiseAddScale <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                        (dArrays_[j].cDField(),
                         tempDev[0].cDField(), coeffs_[i], systemPtr_->basis().nStar());
                    }
                }
                for (int i = 0; i < systemPtr_->dpdmixture().nMonomer(); ++i) 
                {
                    assignReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (systemPtr_->wField(i).cDField(),
                     wArrays_[i].cDField(), systemPtr_->basis().nStar());
                    pointWiseAddScale <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (systemPtr_->wField(i).cDField(),
                     dArrays_[i].cDField(), lambda_, systemPtr_->basis().nStar());
                }

                if (isFlexible_) 
                {
                    for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m)
                    {
                        wCpArrays_[m] = CpHists_[0][m];
                        dCpArrays_[m] = devCpHists_[0][m];
                    }
                    for (int i = 0; i < nHist_; ++i) 
                    {
                        for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m) 
                        {
                            wCpArrays_[m] += coeffs_[i] * ( CpHists_[i+1][m]-
                                                            CpHists_[0][m]);
                            dCpArrays_[m] += coeffs_[i] * ( devCpHists_[i+1][m]-
                                                            devCpHists_[0][m]);
                        }
                    }

                    cellParameters_.clear();
                    for (int m = 0; m < systemPtr_->unitCell().nParameter() ; ++m)
                    {
                        cellParameters_.append(wCpArrays_[m] + lambda_* dCpArrays_[m]);
                    }

                    systemPtr_->unitCell().setParameters(cellParameters_);
                    systemPtr_->dpdmixture().setupUnitCell(systemPtr_->unitCell(), 
                                                           systemPtr_->wavelist());
                    systemPtr_->wavelist().computedKSq(systemPtr_->unitCell());
                }
            }
        }
        
    }
}
}

#endif
