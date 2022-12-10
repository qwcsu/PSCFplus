dpd_solvers_= \
     dpd/solvers/DPDpolymer.cu \
     dpd/solvers/DPDBlock.cu \
     dpd/solvers/DPDpropagator.cu\
     dpd/solvers/DPDmixture.cu\
     dpd/solvers/Joint.cu

dpd_solvers_SRCS=\
     $(addprefix $(SRC_DIR)/, $(dpd_solvers_))
dpd_solvers_OBJS=\
     $(addprefix $(BLD_DIR)/, $(dpd_solvers_:.cu=.o))