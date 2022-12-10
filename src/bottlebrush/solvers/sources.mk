bbcpg_solvers_= \
     bottlebrush/solvers/Bottlebrush.cu

bbcpg_solvers_SRCS=\
     $(addprefix $(SRC_DIR)/, $(bbcpg_solvers_))
dpdpg_solvers_OBJS=\
     $(addprefix $(BLD_DIR)/, $(bbcpg_solvers_:.cu=.o))