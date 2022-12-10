#-----------------------------------------------------------------------
# The copy of this namespace-level makefile in the src/ directory is 
# copied to the bld/serial and bld/parallel directories by the setup
# script to create the copies in those directories. Only the copy in
# the src/ directory is stored in the repository.
#-----------------------------------------------------------------------
# Include makefiles

SRC_DIR_REL =..
include $(SRC_DIR_REL)/config.mk
include $(SRC_DIR)/bottlebrush/include.mk

#-----------------------------------------------------------------------
# Variable definition

BBC_PG1D=$(BLD_DIR)/bottlebrush/bbc_pg1d
BBC_PG2D=$(BLD_DIR)/bottlebrush/bbc_pg2d
BBC_PG3D=$(BLD_DIR)/bottlebrush/bbc_pg3d

BBC_PG_EXE=$(BBC_PG3D_EXE) $(BBC_PG2D_EXE) $(BBC_PG1D_EXE)

#-----------------------------------------------------------------------
# Main targets 

all: $(bbc_OBJS) $(bbc_LIB) $(BBC_PG_EXE)

clean:
	rm -f $(bbc_OBJS) $(bbc_OBJS:.o=.d)
	rm -f $(BBC_PG1D).o $(BBC_PG1D).d
	rm -f $(BBC_PG2D).o $(BBC_PG2D).d
	rm -f $(BBC_PG3D).o $(BBC_PG3D).d
	rm -f $(bbc_LIB)
	rm -f *.o *.d */*.o */*.d
#cd tests; $(MAKE) clean

veryclean:
	$(MAKE) clean
	-rm -f lib*.a

# Executable target

$(BBC_PG1D_EXE): $(BBC_PG1D).o $(BBC_LIBS)
	$(CXX) $(LDFLAGS) -o $(BBC_PG1D_EXE) $(BBC_PG1D).o $(LIBS)

$(BBC_PG2D_EXE): $(BBC_PG2D).o $(BBC_LIBS)
	$(CXX) $(LDFLAGS) -o $(BBC_PG2D_EXE) $(BBC_PG2D).o $(LIBS)

$(BBC_PG3D_EXE): $(BBC_PG3D).o $(BBC_LIBS)
	$(CXX) $(LDFLAGS) -o $(BBC_PG3D_EXE) $(BBC_PG3D).o $(LIBS)

# Short name for executable target (for convenience)
bbc_pg1d:
	$(MAKE) $(BBC_PG1D_EXE)

bbc_pg2d:
	$(MAKE) $(BBC_PG2D_EXE)

bcc_pg3d:
	$(MAKE) $(BBC_PG3D_EXE)

#-----------------------------------------------------------------------
# Include dependency files

-include $(bbc_OBJS:.o=.d)
-include $(pspg_OBJS:.o=.d)
-include $(pscf_OBJS:.o=.d)
-include $(util_OBJS:.o=.d)
-include $(BBC_PG1D).d 
-include $(BBC_PG3D).d 
-include $(BBC_PG3D).d 
