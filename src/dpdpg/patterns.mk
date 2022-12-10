# Local pscf-specific libraries needed in src/pspg
DPDPG_LIBS=$(dpdpg_LIB) $(pspg_LIB) $(pscf_LIB) $(util_LIB)

# All libraries needed in executables built in src/pssp
LIBS=$(DPDPG_LIBS)

# Add paths to Gnu scientific library (GSL)
INCLUDES+=$(GSL_INC)
LIBS+=$(GSL_LIB) 

# Add paths to Cuda FFT library
PSPG_DEFS+=-DPSPG_FFTW -DGPU_OUTER
INCLUDES+=$(CUFFT_INC)
LIBS+=$(CUFFT_LIB)

# Preprocessor macro definitions needed in src/pssp
DEFINES=$(UTIL_DEFS) $(PSCF_DEFS) $(PSPG_DEFS)  $(DPDPG_DEFS)

# Dependencies on build configuration files
MAKE_DEPS= -A$(BLD_DIR)/config.mk
MAKE_DEPS+= -A$(BLD_DIR)/util/config.mk
MAKE_DEPS+= -A$(BLD_DIR)/pscf/config.mk
MAKE_DEPS+= -A$(BLD_DIR)/pspg/config.mk
MAKE_DEPS+= -A$(BLD_DIR)/dpdpg/config.mk

# Pattern rule to compile *.cpp class source files in src/pssp
$(BLD_DIR)/%.o:$(SRC_DIR)/%.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -c -o $@ $<
ifdef MAKEDEP
	$(MAKEDEP) $(INCLUDES) $(DEFINES) $(MAKE_DEPS) -S$(SRC_DIR) -B$(BLD_DIR) $<
endif

# Pattern rule to compile *.cu class source files in src/pssp
$(BLD_DIR)/%.o:$(SRC_DIR)/%.cu
	$(NVXX) $(CPPFLAGS) $(NVXXFLAGS) $(INCLUDES) $(DEFINES) -c -o $@ $<
ifdef MAKEDEP_CUDA
	$(MAKEDEP_CUDA) $(INCLUDES) $(DEFINES) $(MAKE_DEPS) -S$(SRC_DIR) -B$(BLD_DIR) $<
endif

# Pattern rule to compile *.ccu test programs in src/pssp/tests
$(BLD_DIR)/% $(BLD_DIR)/%.o:$(SRC_DIR)/%.ccu $(PSPG_LIBS)
	$(NVXX)  $(NVXXFLAGS) $(INCLUDES) $(DEFINES) -c -o $@ $<
	$(NVXX) $(LDFLAGS) $(INCLUDES) $(DEFINES) -o $(@:.o=) $@ $(LIBS)
ifdef MAKEDEP_CUDA
	$(MAKEDEP_CUDA) $(INCLUDES) $(DEFINES) $(MAKE_DEPS) -S$(SRC_DIR) -B$(BLD_DIR) $<
endif