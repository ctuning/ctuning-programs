# Alchemist low-level plugin for GCC
#
# Author(s): Abdul Wahid Memon
#            Grigori Fursin (http://cTuning.org/lab/people/gfursin)
#
# Started on 2011.09

# Plugin name
PLUGIN_NAME=alchemist-1.0

# Compiler plugin interface version
PLUGIN_ICI_VERSION=gcc-plugin--default

# Plugin name full
PLUGIN_FULL_NAME=$(PLUGIN_ICI_VERSION)--$(PLUGIN_NAME)

# Compiler (GCC >= 4.6)
# Any compiler that generates an ABI compatible with your platform is OK.
# In particular, any GCC can be used to compile the plugin as long as the
# correct headers are used (cf. PLUGIN_INCPATH).
# GCC >= 4.6 for plugins
#GCC_INSTALL=  Add your GCC installation directory
#GCC_SRC=  Add your GCC SRC directory
GCC_INSTALL_INCLUDE=$(GCC_INSTALL)/include
GCC_INSTALL_PLUGINS_DIR=$(shell $(GCC_INSTALL)/bin/gcc -print-file-name=plugin)
GCC_INSTALL_PLUGINS_DIR_INCLUDE=$(GCC_INSTALL_PLUGINS_DIR)/include
GCC_INCLUDE=$(GCC_SRC)/gcc

CC=gcc
CC_SHARED=$(CC) -fPIC -ldl
LD=gcc
LD_SHARED=$(LD) -shared -ldl
CC_OPTS=-O2
LIB_EXT=lib

# GMP
#GMP_INSTALL= Add your GMP installation directory
GMP_INCLUDE=$(GMP_INSTALL)/include

# cJSON
#CJSON_INSTALL= Add your cJSON installation directory
CJSON_LIB=$(CJSON_INSTALL)/$(LIB_EXT)
CJSON_INCLUDE=$(CJSON_INSTALL)/include

# Installation directory for the shared library
INSTALL_DIR=../
INSTALL_DIR_LIB=$(INSTALL_DIR)/$(LIB_EXT)

# All
all: $(PLUGIN_FULL_NAME).so

# Compile shared
$(PLUGIN_FULL_NAME).so: *.c
	$(CC_SHARED) $(CC_OPTS) -c $< -I$(GCC_INSTALL_PLUGINS_DIR_INCLUDE) -I$(GCC_INSTALL_INCLUDE) -I$(GCC_INCLUDE) -I$(GMP_INCLUDE) -I$(CJSON_INCLUDE)
	$(LD_SHARED) -L$(CJSON_LIB) -lcjson -o $@ $(<:.c=.o)
	
# Install
install:
	[ -d $(INSTALL_DIR_LIB) ] || mkdir -p $(INSTALL_DIR_LIB)
	cp -f $(PLUGIN_FULL_NAME).so $(INSTALL_DIR_LIB)

# Clean
clean:
	rm -f *.o
	rm -f $(PLUGIN_FULL_NAME).o
	rm -f $(PLUGIN_FULL_NAME).so
