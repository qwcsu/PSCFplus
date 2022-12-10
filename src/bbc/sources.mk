include $(SRC_DIR)/bbc/solvers/sources.mk

bbc_= \
  $(bbc_solvers_) \
  bbc/System.cu

bbc_SRCS=\
     $(addprefix $(SRC_DIR)/, $(bbc_))
bbc_OBJS=\
     $(addprefix $(BLD_DIR)/, $(bbc_:.cu=.o))

$(bbc_LIB): $(bbc_OBJS)
	$(AR) rcs $(bbc_LIB) $(bbc_OBJS)

