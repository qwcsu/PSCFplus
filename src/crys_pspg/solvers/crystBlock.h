#ifndef CRYST_BLOCK_H
#define CRYST_BLOCK_H

#include "pspg/solvers/Block.h"

using namespace Pscf::Pspg;
using namespace Util;

template <int D>
class crystBlock : public Block<D>
{

public:

    crystBlock();

    ~crystBlock();

};

template <int D>
crystBlock<D>::crystBlock()
{
    std::cout << "hello from crystBlock~\n";
}

template <int D>
crystBlock<D>::~crystBlock()
{
    std::cout << "goodbye from crystBlock~\n";
}

#endif