# include makefiles

SRC_DIR_REL =..
include $(SRC_DIR_REL)/config.mk
include $(SRC_DIR)/dpd/include.mk

#-----------------------------------------------------------------------
# Variable definition

DPD_1D=$(BLD_DIR)/dpd/dpd_1d
DPD_2D=$(BLD_DIR)/dpd/dpd_2d
DPD_3D=$(BLD_DIR)/dpd/dpd_3d

DPD_EXE=$(DPD_3D_EXE) $(DPD_2D_EXE) $(DPD_1D_EXE)

#-----------------------------------------------------------------------
# Main targets 

all: $(dpd_OBJS) $(dpd_LIB) $(DPD_EXE)

clean:
	rm -f $(dpd_OBJS) $(dpd_OBJS:.o=.d)
	rm -f $(DPD_1D).o $(DPD_1D).d
	rm -f $(DPD_2D).o $(DPD_2D).d
	rm -f $(DPD_3D).o $(DPD_3D).d
	rm -f $(dpd_LIB)
	rm -f *.o *.d */*.o */*.d
	#cd tests; $(MAKE) clean

veryclean:
	$(MAKE) clean
	-rm -f lib*.a

# Executable target

$(DPD_1D_EXE): $(DPD_1D).o $(DPD_LIBS)
	$(CXX) $(LDFLAGS) -o $(DPD_1D_EXE) $(DPD_1D).o $(LIBS)

$(DPD_2D_EXE): $(DPD_2D).o $(DPD_LIBS)
	$(CXX) $(LDFLAGS) -o $(DPD_2D_EXE) $(DPD_2D).o $(LIBS)

$(DPD_3D_EXE): $(DPD_3D).o $(DPD_LIBS)
	$(CXX) $(LDFLAGS) -o $(DPD_3D_EXE) $(DPD_3D).o $(LIBS)

# Short name for executable target (for convenience)
dpd_1d:
	$(MAKE) $(DPD_1D_EXE)

dpd_2d:
	$(MAKE) $(DPD_2D_EXE)

dpd_3d:
	$(MAKE) $(DPD_3D_EXE)

#-----------------------------------------------------------------------
# Include dependency files

-include $(dpd_OBJS:.o=.d)
-include $(pspg_OBJS:.o=.d)
-include $(pscf_OBJS:.o=.d)
-include $(util_OBJS:.o=.d)
-include $(DPD_1D).d 
-include $(DPD_2D).d 
-include $(DPD_3D).d 
