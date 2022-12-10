# include makefiles

SRC_DIR_REL =..
include $(SRC_DIR_REL)/config.mk
include $(SRC_DIR)/dpdpg/include.mk

#-----------------------------------------------------------------------
# Variable definition

DPD_PG1D=$(BLD_DIR)/dpdpg/dpd_pg1d
DPD_PG2D=$(BLD_DIR)/dpdpg/dpd_pg2d
DPD_PG3D=$(BLD_DIR)/dpdpg/dpd_pg3d
DPD_PGPHASES=$(BLD_DIR)/dpdpg/dpd_pgphases

DPD_PG_EXE=$(DPD_PG3D_EXE) $(DPD_PG2D_EXE) $(DPD_PG1D_EXE) $(DPD_PGPHASES_EXE)

#-----------------------------------------------------------------------
# Main targets 

all: $(dpdpg_OBJS) $(dpdpg_LIB) $(DPD_PG_EXE)

clean:
	rm -f $(dpdpg_OBJS) $(dpdpg_OBJS:.o=.d)
	rm -f $(DPD_PG1D).o $(DPD_PG1D).d
	rm -f $(DPD_PG3D).o $(DPD_PG2D).d
	rm -f $(DPD_PG3D).o $(DPD_PG3D).d
	rm -f $(DPD_PGPHASES).o $(DPD_PGPHASES).d
	rm -f $(dpdpg_LIB)
	rm -f *.o *.d */*.o */*.d
	#cd tests; $(MAKE) clean

veryclean:
	$(MAKE) clean
	-rm -f lib*.a

# Executable target

$(DPD_PG1D_EXE): $(DPD_PG1D).o $(DPDPG_LIBS)
	$(CXX) $(LDFLAGS) -o $(DPD_PG1D_EXE) $(DPD_PG1D).o $(LIBS)

$(DPD_PG2D_EXE): $(DPD_PG2D).o $(DPDPG_LIBS)
	$(CXX) $(LDFLAGS) -o $(DPD_PG2D_EXE) $(DPD_PG2D).o $(LIBS)

$(DPD_PG3D_EXE): $(DPD_PG3D).o $(DPDPG_LIBS)
	$(CXX) $(LDFLAGS) -o $(DPD_PG3D_EXE) $(DPD_PG3D).o $(LIBS)
	
$(DPD_PGPHASES_EXE): $(DPD_PGPHASES).o $(DPDPG_LIBS)
	$(CXX) $(LDFLAGS) -o $(DPD_PGPHASES_EXE) $(DPD_PGPHASES).o $(LIBS)

# Short name for executable target (for convenience)
dpd_pg1d:
	$(MAKE) $(DPD_PG1D_EXE)

dpd_pg2d:
	$(MAKE) $(DPD_PG2D_EXE)

dpd_pg3d:
	$(MAKE) $(DPD_PG3D_EXE)
	
dpd_pgphases:
	$(MAKE) $(DPD_PGPHASES_EXE)

#-----------------------------------------------------------------------
# Include dependency files

-include $(dpdpg_OBJS:.o=.d)
-include $(pspg_OBJS:.o=.d)
-include $(pscf_OBJS:.o=.d)
-include $(util_OBJS:.o=.d)
-include $(DPD_PG1D).d 
-include $(DPD_PG3D).d 
-include $(DPD_PG3D).d 
-include $(DPD_PGPHASES).d 
