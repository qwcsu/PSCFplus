dpd_iterator_= \
  dpd/iterator/AmIterator.cu \
  dpd/iterator/Iterator.cu 

  

dpd_iterator_SRCS=\
	  $(addprefix $(SRC_DIR)/, $(dpd_iterator_))
dpd_iterator_OBJS=\
	  $(addprefix $(BLD_DIR)/, $(dpd_iterator_:.cu=.o))
