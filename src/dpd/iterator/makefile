#--------------------------------------------------------------------
# Include makefiles

SRC_DIR_REL =../..
include $(SRC_DIR_REL)/config.mk
include $(SRC_DIR)/pspg/include.mk
include $(SRC_DIR)/dpd/include.mk

#--------------------------------------------------------------------
# Main targets 

all: $(dpd_iterator_OBJS) 

includes:
	echo $(INCLUDES)

clean:
	rm -f $(dpd_iterator_OBJS) $(dpd_iterator_OBJS:.o=.d) 

#--------------------------------------------------------------------
# Include dependency files

-include $(dpd_OBJS:.o=.d)
