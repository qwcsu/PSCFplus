# include $(SRC_DIR)/dpdpg/iterator/sources.mk
include $(SRC_DIR)/bottlebrush/solvers/sources.mk

bbcpg_= \
     $(bbcpg_solvers_) \
     bottlebrush/System.cu

bbcpg_SRCS=\
     $(addprefix $(SRC_DIR)/, $(bbcpg_))
bbcpg_OBJS=\
     $(addprefix $(BLD_DIR)/, $(bbcpg_:.cu=.o))

$(bbcpg_LIB): $(bbcpg_OBJS)
	$(AR) rcs $(bbcpg_LIB) $(bbcpg_OBJS)