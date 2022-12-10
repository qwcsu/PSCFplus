#ifndef PSPC_FIELD_IO_TPP
#define PSPC_FIELD_IO_TPP

/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "FieldIo.h"

#include <pscf/crystal/shiftToMinimum.h>
#include <pscf/mesh/MeshIterator.h>
#include <pscf/math/IntVec.h>

#include <util/format/Str.h>
#include <util/format/Int.h>
#include <util/format/Dbl.h>

#include <iomanip>
#include <string>

namespace Pscf {
namespace Pspc
{

   using namespace Util;

   /*
   * Constructor.
   */
   template <int D>
   FieldIo<D>::FieldIo()
    : unitCellPtr_(0),
      meshPtr_(0),
      fftPtr_(0),
      groupNamePtr_(0),
      basisPtr_(0),
      fileMasterPtr_()
   {}

   /*
   * Destructor.
   */
   template <int D>
   FieldIo<D>::~FieldIo()
   {}

   /*
   * Get and store addresses of associated objects.
   */
   template <int D>
   void FieldIo<D>::associate(UnitCell<D>& unitCell,
                             Mesh<D>& mesh,
                             FFT<D>& fft,
                             std::string& groupName,
                             Basis<D>& basis,
                             FileMaster& fileMaster)
   {
      unitCellPtr_ = &unitCell;
      meshPtr_ = &mesh;
      groupNamePtr_ = &groupName;
      basisPtr_ = &basis;
      fftPtr_ = &fft;
      fileMasterPtr_ = &fileMaster;
   }
  
   template <int D>
   void FieldIo<D>::readFieldsBasis(std::istream& in, 
                                    DArray< DArray<double> >& fields)
   {
      int nMonomer = fields.capacity();
      UTIL_CHECK(nMonomer > 0);

      // Read header
      FieldIo<D>::readFieldHeader(in);
      std::string label;
      in >> label;
      UTIL_CHECK(label == "N_star");
      int nStarIn;
      in >> nStarIn;
      UTIL_CHECK(nStarIn > 0);

      // Initialize all field array elements to zero
      int i, j;
      int nStarCapacity;
      for (j = 0; j < nMonomer; ++j) {
         nStarCapacity = fields[j].capacity();
         UTIL_CHECK(nStarCapacity >= nStarIn);
         for (i = 0; i < nStarCapacity; ++i) {
            fields[j][i] = 0.0;
         }
      }

      DArray<double> temp;
      temp.allocate(nMonomer);

      // Loop over stars to read field components
      IntVec<D> waveIn, waveBz, waveDft;
      int waveId, starId, nWaveVectors;
      bool waveExists;
      for (i = 0; i < nStarIn; ++i) {

         // Read components for different monomers
         for (j = 0; j < nMonomer; ++j) {
            in >> temp [j];
         }

         // Read characteristic wave and number of wavectors in star.
         in >> waveIn;
         in >> nWaveVectors;

         // Check if waveIn is in first Brillouin zone (FBZ) for the mesh.
         waveBz = shiftToMinimum(waveIn, mesh().dimensions(), unitCell());
         waveExists = (waveIn == waveBz);

         // If wave is in FBZ, find in basis and set field components
         if (waveExists) {
            waveDft = waveBz;
            mesh().shift(waveDft);
            waveId = basis().waveId(waveDft);
            starId = basis().wave(waveId).starId;
            UTIL_CHECK(basis().star(starId).waveBz == waveBz);
            if (!basis().star(starId).cancel) {
               for (j = 0; j < nMonomer; ++j) {
                  fields[j][starId] = temp [j];
               }
            }
         }

      }

   }
   
   template <int D>
   void FieldIo<D>::readFieldsBasis(std::string filename, 
                              DArray<DArray<double> >& fields)
   {
       std::ifstream file;
       fileMaster().openInputFile(filename, file);
       readFieldsBasis(file, fields);
       file.close();
   }

   template <int D>
   void 
   FieldIo<D>::writeFieldsBasis(std::ostream &out, 
                                DArray<DArray<double> > const &  fields)
   {
      int nMonomer = fields.capacity();
      UTIL_CHECK(nMonomer > 0);

      // Write header
      writeFieldHeader(out, nMonomer);
      int nStar = basis().nStar();
      int nBasis = basis().nBasis();
      out << "N_star       " << std::endl 
          << "             " << nBasis << std::endl;

      // Write fields
      for (int i = 0; i < nStar; ++i) {
         if (!basis().star(i).cancel) {
            for (int j = 0; j < nMonomer; ++j) {
               out << Dbl(fields[j][i], 20, 10);
            }
            out << "   ";
            for (int j = 0; j < D; ++j) {
               out << Int(basis().star(i).waveBz[j], 5);
            } 
            out << Int(basis().star(i).size, 5) << std::endl;
         }
      }

   }

   template <int D>
   void FieldIo<D>::writeFieldsBasis(std::string filename, 
                                     DArray<DArray<double> > const & fields)
   {
       std::ofstream file;
       fileMaster().openOutputFile(filename, file);
       writeFieldsBasis(file, fields);
       file.close();
   }

   template <int D>
   void FieldIo<D>::readFieldsRGrid(std::istream &in,
                                    DArray<RField<D> >& fields)
   {
      int nMonomer = fields.capacity();
      UTIL_CHECK(nMonomer > 0);

      FieldIo<D>::readFieldHeader(in);

      std::string label;
      in >> label;
      UTIL_CHECK(label == "ngrid");
      IntVec<D> nGrid;
      in >> nGrid;
      UTIL_CHECK(nGrid == mesh().dimensions());

      DArray<RField<D> > temp;
      temp.allocate(nMonomer);
      for (int i = 0; i < nMonomer; ++i) {
         temp[i].allocate(mesh().dimensions());
      }

      // Read Fields;
      MeshIterator<D> itr(mesh().dimensions());
      for (itr.begin(); !itr.atEnd(); ++itr) {
         for (int i = 0; i < nMonomer; ++i) {
            in  >> std::setprecision(15) >> temp[i][itr.rank()];
         }
      }

      int p = 0;
      int q = 0;
      int r = 0;
      int s = 0;
      int n1 =0;
      int n2 =0;
      int n3 =0;

      if (D==3) {
         while (n1 < mesh().dimension(0)) {
            q = p;
            n2 = 0;
            while (n2 < mesh().dimension(1)) {
               r =q;
               n3 = 0;
               while (n3 < mesh().dimension(2)) {
                  for (int i = 0; i < nMonomer; ++i) {
                     fields[i][s] = temp[i][r];
                  }
                  r = r + (mesh().dimension(0) * mesh().dimension(1));
                  ++s;
                  ++n3;              
               } 
               q = q + mesh().dimension(0);
               ++n2;
            } 
            ++n1;
            ++p;        
         }
      }

      else if (D==2) {
         while (n1 < mesh().dimension(0)) {
            r =q; 
            n2 = 0;
            while (n2 < mesh().dimension(1)) {
               for (int i = 0; i < nMonomer; ++i) {
                  fields[i][s] = temp[i][r];
               }   
               r = r + (mesh().dimension(0));
               ++s;
               ++n2;    
            }   
            ++q;
            ++n1;
         }   
      } 

      else if (D==1) {

         while (n1 < mesh().dimension(0)) {
            for (int i = 0; i < nMonomer; ++i) {
               fields[i][s] = temp[i][r];
            }   
            ++r;
            ++s;
            ++n1;    
         }   
      } 

      else{
         std::cout << "Invalid Dimensions";
      }

   }

   template <int D>
   void FieldIo<D>::readFieldsRGrid(std::string filename, 
                              DArray< RField<D> >& fields)
   {
      std::ifstream file;
      fileMaster().openInputFile(filename, file);
      readFieldsRGrid(file, fields);
      file.close();
   }

   template <int D>
   void FieldIo<D>::writeFieldsRGrid(std::ostream &out,
                                     DArray<RField<D> > const& fields)
   {
      int nMonomer = fields.capacity();
      UTIL_CHECK(nMonomer > 0);

      writeFieldHeader(out, nMonomer);
      out << "ngrid" <<  std::endl
          << "           " << mesh().dimensions() << std::endl;

      DArray<RField<D> > temp;
      temp.allocate(nMonomer);
      for (int i = 0; i < nMonomer; ++i) {
         temp[i].allocate(mesh().dimensions());
      } 

      int p = 0; 
      int q = 0; 
      int r = 0; 
      int s = 0; 
      int n1 =0;
      int n2 =0;
      int n3 =0;

      if (D==3) {
         while (n3 < mesh().dimension(2)) {
            q = p; 
            n2 = 0; 
            while (n2 < mesh().dimension(1)) {
               r =q;
               n1 = 0; 
               while (n1 < mesh().dimension(0)) {
                  for (int i = 0; i < nMonomer; ++i) {
                     temp[i][s] = fields[i][r];
                  }    
                  r = r + (mesh().dimension(1) * mesh().dimension(2));
                  ++s; 
                  ++n1;     
               }    
               q = q + mesh().dimension(2);
               ++n2;
            }    
            ++n3;
            ++p;     
         }    
      }
      else if (D==2) {
         while (n2 < mesh().dimension(1)) {
            r =q;
            n1 = 0;
            while (n1 < mesh().dimension(0)) {
               for (int i = 0; i < nMonomer; ++i) {
                  temp[i][s] = fields[i][r];
               }
               r = r + (mesh().dimension(1));
               ++s;
               ++n1;
            }
            ++q;
            ++n2;
         }
      }
      else if (D==1) {
         while (n1 < mesh().dimension(0)) {
            for (int i = 0; i < nMonomer; ++i) {
               temp[i][s] = fields[i][r];
            }
            ++r;
            ++s;
            ++n1;
         }
      } else {
         std::cout << "Invalid Dimensions";
      }

      // Write fields
      MeshIterator<D> itr(mesh().dimensions());
      for (itr.begin(); !itr.atEnd(); ++itr) {
         // out << Int(itr.rank(), 5);
         for (int j = 0; j < nMonomer; ++j) {
            out << "  " << Dbl(temp[j][itr.rank()], 18, 15);
         }
         out << std::endl;
      }

   }

   template <int D>
   void FieldIo<D>::writeFieldsRGrid(std::string filename, 
                                     DArray< RField<D> > const & fields)
   {
      std::ofstream file;
      fileMaster().openOutputFile(filename, file);
      writeFieldsRGrid(file, fields);
      file.close();
   }

   template <int D>
   void FieldIo<D>::readFieldsKGrid(std::istream &in,
                                    DArray<RFieldDft<D> >& fields)
   {
      int nMonomer = fields.capacity();
      UTIL_CHECK(nMonomer > 0);

      // Read header
      readFieldHeader(in);
      std::string label;
      in >> label;
      UTIL_CHECK(label == "ngrid");
      IntVec<D> nGrid;
      in >> nGrid;
      UTIL_CHECK(nGrid == mesh().dimensions());

      // Read Fields;
      int idum;
      MeshIterator<D> itr(mesh().dimensions());
      for (itr.begin(); !itr.atEnd(); ++itr) {
         in >> idum;
         for (int i = 0; i < nMonomer; ++i) {
            for (int j = 0; j < 2; ++j) {
               in >> fields[i][itr.rank()][j];
            }
         }
      }
   }

   template <int D>
   void FieldIo<D>::readFieldsKGrid(std::string filename, 
                                    DArray< RFieldDft<D> >& fields)
   {
      std::ifstream file;
      fileMaster().openInputFile(filename, file);
      readFieldsKGrid(file, fields);
      file.close();
   }

   template <int D>
   void FieldIo<D>::writeFieldsKGrid(std::ostream &out,
                                     DArray<RFieldDft<D> > const& fields)
   {
      int nMonomer = fields.capacity();
      UTIL_CHECK(nMonomer > 0);

      // Write header
      writeFieldHeader(out, nMonomer);
      out << "ngrid" << std::endl 
          << "               " << mesh().dimensions() << std::endl;

      // Write fields
      MeshIterator<D> itr(mesh().dimensions());
      for (itr.begin(); !itr.atEnd(); ++itr) {
         out << Int(itr.rank(), 5);
         for (int j = 0; j < nMonomer; ++j) {
               out << "  " << Dbl(fields[j][itr.rank()][0], 18, 11)
                   << Dbl(fields[j][itr.rank()][1], 18, 11);
         }
         out << std::endl;
      }
   }

   template <int D>
   void FieldIo<D>::writeFieldsKGrid(std::string filename, 
                                    DArray< RFieldDft<D> > const& fields)
   {
      std::ofstream file;
      fileMaster().openOutputFile(filename, file);
      writeFieldsKGrid(file, fields);
      file.close();
   }

   template <int D>
   void FieldIo<D>::readFieldHeader(std::istream& in) 
   {
      std::string label;

      in >> label;
      UTIL_CHECK(label == "format");
      int ver1, ver2;
      in >> ver1 >> ver2;
 
      in >> label;
      UTIL_CHECK(label == "dim");
      int dim;
      in >> dim;
      UTIL_CHECK(dim == D);

      readUnitCellHeader(in, unitCell());

      in >> label;
      UTIL_CHECK(label == "group_name");
      std::string groupName;
      in >> groupName;

      in >> label;
      UTIL_CHECK(label == "N_monomer");
      int nMonomer;
      in >> nMonomer;
      UTIL_CHECK(nMonomer > 0);
   }

   template <int D>
   void FieldIo<D>::writeFieldHeader(std::ostream &out, int nMonomer) const
   {
      out << "format  1   0" <<  std::endl;
      out << "dim" <<  std::endl 
          << "          " << D << std::endl;
      writeUnitCellHeader(out, unitCell()); 
      out << "group_name" << std::endl 
          << "          " << groupName() <<  std::endl;
      out << "N_monomer"  << std::endl 
          << "          " << nMonomer << std::endl;
   }

   template <int D>
   void FieldIo<D>::convertBasisToKGrid(DArray<double> const& in, 
                                        RFieldDft<D>& out)
   {
      // Create Mesh<D> with dimensions of DFT Fourier grid.
      Mesh<D> dftMesh(out.dftDimensions());

      typename Basis<D>::Star const* starPtr; // pointer to current star
      typename Basis<D>::Wave const* wavePtr; // pointer to current wave
      std::complex<double> component;         // coefficient for star
      std::complex<double> coeff;             // coefficient for wave
      IntVec<D> indices;                      // dft grid indices of wave
      int rank;                               // dft grid rank of wave
      int is;                                 // star index
      int iw;                                 // wave index

      // Initialize all dft coponents to zero
      for (rank = 0; rank < dftMesh.size(); ++rank) {
         out[rank][0] = 0.0;
         out[rank][1] = 0.0;
      }

      // Loop over stars, skipping cancelled stars
      is = 0;
      while (is < basis().nStar()) {
         starPtr = &(basis().star(is));

         if (starPtr->cancel) {
            ++is;
            continue;
         }

         if (starPtr->invertFlag == 0) {

            // Make complex coefficient for star basis function
            component = std::complex<double>(in[is], 0.0);

            // Loop over waves in closed star
            for (iw = starPtr->beginId; iw < starPtr->endId; ++iw) {
               wavePtr = &basis().wave(iw);
               if (!wavePtr->implicit) {
                  coeff = component*(wavePtr->coeff);
                  indices = wavePtr->indicesDft;    
                  rank = dftMesh.rank(indices);
                  out[rank][0] = coeff.real();
                  out[rank][1] = coeff.imag();
               }
            }
            ++is;

         } else
         if (starPtr->invertFlag == 1) {

            // Make complex component for first star
            component = std::complex<double>(in[is], -in[is+1]);
            component /= sqrt(2.0);

            // Loop over waves in first star
            starPtr = &(basis().star(is));
            for (iw = starPtr->beginId; iw < starPtr->endId; ++iw) {
               wavePtr = &basis().wave(iw);
               if (!(wavePtr->implicit)) {
                  coeff = component*(wavePtr->coeff);
                  indices = wavePtr->indicesDft;    
                  rank = dftMesh.rank(indices);
                  out[rank][0] = coeff.real();
                  out[rank][1] = coeff.imag();
               }
            }

            // Loop over waves in second star
            starPtr = &(basis().star(is+1));
            UTIL_CHECK(starPtr->invertFlag == -1);
            component = conj(component);
            for (iw = starPtr->beginId; iw < starPtr->endId; ++iw) {
               wavePtr = &basis().wave(iw);
               if (!(wavePtr->implicit)) {
                  coeff = component*(wavePtr->coeff);
                  indices = wavePtr->indicesDft;
                  rank = dftMesh.rank(indices);
                  out[rank][0] = coeff.real();
                  out[rank][1] = coeff.imag();
               }
            }

            // Increment is by 2 (two stars were processed)
            is += 2;

         } else {
 
            UTIL_THROW("Invalid invertFlag value");
  
         }

      }

   }

   template <int D>
   void FieldIo<D>::convertKGridToBasis(RFieldDft<D> const& in, 
                                        DArray<double>& out)
   {
      // Create Mesh<D> with dimensions of DFT Fourier grid.
      Mesh<D> dftMesh(in.dftDimensions());

      typename Basis<D>::Star const* starPtr;  // pointer to current star
      typename Basis<D>::Wave const* wavePtr;  // pointer to current wave
      std::complex<double> component;          // coefficient for star
      IntVec<D> indices;                       // dft grid indices of wave
      int rank;                                // dft grid rank of wave
      int is;                                  // star index
      int iw;                                  // wave id, within star 
      bool isImplicit;

      // Initialize all components to zero
      for (is = 0; is < basis().nStar(); ++is) {
         out[is] = 0.0;
      }

      // Loop over stars
      is = 0;
      while (is < basis().nStar()) {
         starPtr = &(basis().star(is));

         if (starPtr->cancel) {
            ++is;
            continue;
         }

         if (starPtr->invertFlag == 0) {

            // Choose a characteristic wave that is not implicit.
            // Start with the first, alternately searching from
            // the beginning and end of star.
            isImplicit = true;
            iw = 0;
            while (isImplicit) {
                UTIL_CHECK(iw <= (starPtr->size)/2);
                wavePtr = &basis().wave(starPtr->beginId + iw);
                if (wavePtr->implicit) {
                   wavePtr = &basis().wave(starPtr->endId - 1 - iw);
                }
                isImplicit = wavePtr->implicit;
                ++iw;
            }
            UTIL_CHECK(wavePtr->starId == is);
            indices = wavePtr->indicesDft;
            rank = dftMesh.rank(indices);

            // Compute component value
            component = std::complex<double>(in[rank][0], in[rank][1]);
            component /= wavePtr->coeff;
            UTIL_CHECK(abs(component.imag()) < 1.0E-4);
            out[is] = component.real();
            ++is;

         } else
         if (starPtr->invertFlag == 1) {

            // Identify a characteristic wave that is not implicit:
            // Either first wave of 1st star or last wave of 2nd star.
            wavePtr = &basis().wave(starPtr->beginId);
            if (wavePtr->implicit) {
               starPtr = &(basis().star(is+1));
               UTIL_CHECK(starPtr->invertFlag == -1);
               wavePtr = &basis().wave(starPtr->endId-1);
               UTIL_CHECK(!(wavePtr->implicit));
            } 
            indices = wavePtr->indicesDft;
            rank = dftMesh.rank(indices);

            // Compute component value
            component = std::complex<double>(in[rank][0], in[rank][1]);
            UTIL_CHECK(abs(wavePtr->coeff) > 1.0E-8);
            component /= wavePtr->coeff;
            component *= sqrt(2.0);
            out[is] = component.real();
            out[is+1] = -component.imag();

            is += 2;
         } else {
            UTIL_THROW("Invalid invertFlag value");
         }

      } //  loop over star index is
   }

   template <int D>
   void FieldIo<D>::convertBasisToKGrid(DArray< DArray <double> >& in,
                                        DArray< RFieldDft<D> >& out)
   {
      UTIL_ASSERT(in.capacity() == out.capacity());
      int n = in.capacity();
      for (int i = 0; i < n; ++i) {
         convertBasisToKGrid(in[i], out[i]);
      }
   }

   template <int D>
   void FieldIo<D>::convertKGridToBasis(DArray< RFieldDft<D> >& in,
                                        DArray< DArray <double> > & out)
   {
      UTIL_ASSERT(in.capacity() == out.capacity());
      int n = in.capacity();
      for (int i = 0; i < n; ++i) {
         convertKGridToBasis(in[i], out[i]);
      }
   }

   template <int D>
   void 
   FieldIo<D>::convertBasisToRGrid(DArray< DArray <double> >& in,
                                   DArray< RField<D> >& out)
   {
      UTIL_ASSERT(in.capacity() == out.capacity());
      checkWorkDft();

      int n = in.capacity();
      for (int i = 0; i < n; ++i) {
         convertBasisToKGrid(in[i], workDft_);
         fft().inverseTransform(workDft_, out[i]);
      }
   }

   template <int D>
   void 
   FieldIo<D>::convertRGridToBasis(DArray< RField<D> >& in,
                                   DArray< DArray <double> > & out)
   {
      UTIL_ASSERT(in.capacity() == out.capacity());
      checkWorkDft();

      int n = in.capacity();
      for (int i = 0; i < n; ++i) {
         fft().forwardTransform(in[i], workDft_);
         convertKGridToBasis(workDft_, out[i]);
      }
   }

   template <int D>
   void FieldIo<D>::checkWorkDft()
   {
      if (!workDft_.isAllocated()) {
         workDft_.allocate(mesh().dimensions());
      } else {
         UTIL_CHECK(workDft_.meshDimensions() == fft().meshDimensions());
      }
   }

} // namespace Pspc
} // namespace Pscf
#endif
