bbc_solvers_= \
     bbc/solvers/Bottlebrushblock.cu\
     bbc/solvers/Mixture.cu

bbc_solvers_SRCS=\
     $(addprefix $(SRC_DIR)/, $(bbc_solvers_))
bbc_solvers_OBJS=\
     $(addprefix $(BLD_DIR)/, $(bbc_solvers_:.cu=.o))