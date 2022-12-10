include $(SRC_DIR)/dpd/solvers/sources.mk
include $(SRC_DIR)/dpd/iterator/sources.mk

dpd_= dpd/System.cu  \
     $(dpd_solvers_) \
     $(dpd_iterator_) 

dpd_SRCS=\
     $(addprefix $(SRC_DIR)/, $(dpd_))
dpd_OBJS=\
     $(addprefix $(BLD_DIR)/, $(dpd_:.cu=.o))

$(dpd_LIB): $(dpd_OBJS)
	$(AR) rcs $(dpd_LIB) $(dpd_OBJS)