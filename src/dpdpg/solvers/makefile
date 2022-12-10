# Include makefiles

SRC_DIR_REL =../..
include $(SRC_DIR_REL)/config.mk
include $(BLD_DIR)/util/config.mk
include $(BLD_DIR)/pscf/config.mk
include $(BLD_DIR)/pspg/config.mk
include $(SRC_DIR)/pspg/patterns.mk
include $(SRC_DIR)/util/sources.mk
include $(SRC_DIR)/pscf/sources.mk
include $(SRC_DIR)/pspg/sources.mk
include $(SRC_DIR)/dpdpg/sources.mk

#-----------------------------------------------------------------------
# Main targets 

all: $(dpdpg_solvers_OBJS) 

clean:
	rm -f $(dpdpg_solvers_OBJS) $(dpdpg_solvers_OBJS:.o=.d)

veryclean:
	$(MAKE) clean
	-rm -f *.o 
	-rm -f *.d 

#-----------------------------------------------------------------------
# Include dependency files

-include $(dpdpg_OBJS:.o=.d)
-include $(pspg_OBJS:.o=.d)
-include $(pscf_OBJS:.o=.d)
-include $(util_OBJS:.o=.d)
-include $(PSCF_FD).d 
