/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "Vertex.h"
#include "BlockDescriptor.h"
#include "DPDBlockDescriptor.h"
#include "JointDescriptor.h"
#include <util/global.h>

namespace Pscf
{ 

   Vertex::Vertex()
    : inPropagatorIds_(),
      outPropagatorIds_(),
      id_(-1)
   {}

   Vertex::~Vertex()
   {}

   void Vertex::setId(int id)
   {  id_ = id; }

   void Vertex::addBlock(const BlockDescriptor& block)
   {
      // Preconditions
      if (id_ < 0) {
         UTIL_THROW("Negative vertex id");
      }
      if (block.id() < 0) {
         UTIL_THROW("Negative block id");
      }
      if (block.vertexId(0) == block.vertexId(1)) {
         UTIL_THROW("Error: Equal vertex indices in block");
      }

      Pair<int> propagatorId;
      propagatorId[0] = block.id();
      if (block.vertexId(0) == id_) {
         propagatorId[1] = 0;
         outPropagatorIds_.append(propagatorId);
         propagatorId[1] = 1;
         inPropagatorIds_.append(propagatorId);
      } else
      if (block.vertexId(1) == id_) {
         propagatorId[1] = 1;
         outPropagatorIds_.append(propagatorId);
         propagatorId[1] = 0;
         inPropagatorIds_.append(propagatorId);
      } else {
         UTIL_THROW("Neither block vertex id matches this vertex");
      }
   }

   void Vertex::addBlock(const DPDBlockDescriptor& block)
   {
      // Preconditions
      if (id_ < 0) {
         UTIL_THROW("Negative vertex id");
      }
      if (block.id() < 0) {
         UTIL_THROW("Negative block id");
      }
      if (block.vertexId(0) == block.vertexId(1)) {
         UTIL_THROW("Error: Equal vertex indices in block");
      }

      Pair<int> propagatorId;
      propagatorId[0] = block.id();
      if (block.vertexId(0) == id_) {
         propagatorId[1] = 0;
         outPropagatorIds_.append(propagatorId);
         propagatorId[1] = 1;
         inPropagatorIds_.append(propagatorId);
      } else
      if (block.vertexId(1) == id_) {
         propagatorId[1] = 1;
         outPropagatorIds_.append(propagatorId);
         propagatorId[1] = 0;
         inPropagatorIds_.append(propagatorId);
      } else {
         UTIL_THROW("Neither block vertex id matches this vertex");
      }
   }

   void Vertex::addJoint(const JointDescriptor& joint)
   {
      // Preconditions
      if (id_ < 0) {
         UTIL_THROW("Negative vertex id");
      }
      if (joint.id() < 0) {
         UTIL_THROW("Negative joint id");
      }
      if (joint.vertexId(0) == joint.vertexId(1)) {
         UTIL_THROW("Error: Equal vertex indices in joint");
      }

      Pair<int> propagatorId;
      propagatorId[0] = joint.id();
      if (joint.vertexId(0) == id_) {
         propagatorId[1] = 0;
         outPropagatorIds_.append(propagatorId);
         propagatorId[1] = 1;
         inPropagatorIds_.append(propagatorId);
      } else
      if (joint.vertexId(1) == id_) {
         propagatorId[1] = 1;
         outPropagatorIds_.append(propagatorId);
         propagatorId[1] = 0;
         inPropagatorIds_.append(propagatorId);
      } else {
         UTIL_THROW("Neither block vertex id matches this vertex");
      }
   }


} 
