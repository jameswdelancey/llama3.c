CC ?= gcc
CFLAGS ?= -O3 -Ofast -march=native -std=c11 -Wall -Wextra
LDFLAGS ?=
LIBS ?=

USE_OPENMP ?= 1

SRC := run.c
TARGET := run

ifeq ($(USE_OPENMP),1)
CFLAGS += -fopenmp
endif

ifdef USE_MKL
CFLAGS += -DLLAMA_USE_MKL
ifdef MKLROOT
CFLAGS += -I$(MKLROOT)/include
LDFLAGS += -L$(MKLROOT)/lib -L$(MKLROOT)/lib/intel64 -L$(MKLROOT)/lib/intel64_lin
endif
LIBS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -ldl
endif

ifdef USE_OPENBLAS
CFLAGS += -DLLAMA_USE_OPENBLAS
ifdef OPENBLAS_ROOT
CFLAGS += -I$(OPENBLAS_ROOT)/include
LDFLAGS += -L$(OPENBLAS_ROOT)/lib
endif
LIBS += -lopenblas
endif

ifdef USE_ACCELERATE
CFLAGS += -DLLAMA_USE_ACCELERATE
LIBS += -framework Accelerate
endif

LIBS += -lm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
