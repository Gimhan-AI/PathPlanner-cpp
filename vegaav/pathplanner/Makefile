HOME_TREE := ../../

MODULE_NAME := $(notdir $(CURDIR))

include $(HOME_TREE)/mak_def.inc

CXXSRC += main.cpp
C_INCLUDES += -I$(CURDIR)/../../sdk/include -I$(CURDIR)/../../sdk/src `pkg-config --cflags opencv4`

EXTRA_OBJ := 
LD_LIBS += -lstdc++ -lpthread -lm `pkg-config --libs opencv4`

all: build_app

include $(HOME_TREE)/mak_common.inc

clean: clean_app