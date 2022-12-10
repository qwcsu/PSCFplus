#--------------------------------------------------------------------
# Include makefiles

SRC_DIR_REL =../..
include $(SRC_DIR_REL)/config.mk
include $(SRC_DIR)/pspg/include.mk
include $(SRC_DIR)/dpdpg/include.mk

#--------------------------------------------------------------------
# Main targets 

all: $(dpdpg_iterator_OBJS) 

includes:
	echo $(INCLUDES)

clean:
	rm -f $(dpdpg_iterator_OBJS) $(dpdpg_iterator_OBJS:.o=.d) 

#--------------------------------------------------------------------
# Include dependency files

-include $(dpdpg_OBJS:.o=.d)
