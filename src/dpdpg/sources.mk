include $(SRC_DIR)/dpdpg/iterator/sources.mk
include $(SRC_DIR)/dpdpg/solvers/sources.mk

dpdpg_= \
     $(dpdpg_solvers_) \
     $(dpdpg_iterator_) \
     dpdpg/System.cu

dpdpg_SRCS=\
     $(addprefix $(SRC_DIR)/, $(dpdpg_))
dpdpg_OBJS=\
     $(addprefix $(BLD_DIR)/, $(dpdpg_:.cu=.o))

$(dpdpg_LIB): $(dpdpg_OBJS)
	$(AR) rcs $(dpdpg_LIB) $(dpdpg_OBJS)