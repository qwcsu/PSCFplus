dpdpg_solvers_= \
     dpdpg/solvers/DPDdiblock_3m.cu\
     dpdpg/solvers/DPDdiblock.cu\
     dpdpg/solvers/Block.cu  

dpdpg_solvers_SRCS=\
     $(addprefix $(SRC_DIR)/, $(dpdpg_solvers_))
dpdpg_solvers_OBJS=\
     $(addprefix $(BLD_DIR)/, $(dpdpg_solvers_:.cu=.o))