pspg_field_= \
  pspg/field/FFT.cu \
  pspg/field/fct.cu \
  pspg/field/FFTBatched.cu \
  pspg/field/FieldIo.cu 

pspg_field_SRCS=\
	  $(addprefix $(SRC_DIR)/, $(pspg_field_))
pspg_field_OBJS=\
	  $(addprefix $(BLD_DIR)/, $(pspg_field_:.cu=.o))

