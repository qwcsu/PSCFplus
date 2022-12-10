dpdpg_iterator_= \
  dpdpg/iterator/AmIterator.cu \
  dpdpg/iterator/Iterator.cu 

  

dpdpg_iterator_SRCS=\
	  $(addprefix $(SRC_DIR)/, $(dpdpg_iterator_))
dpdpg_iterator_OBJS=\
	  $(addprefix $(BLD_DIR)/, $(dpdpg_iterator_:.cu=.o))
