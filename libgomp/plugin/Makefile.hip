CC ?= gcc
FC ?= gfortran

#CFLAGS  ?= $(shell hipconfig --cpp_config) -I../ -I../../include \
#    -Wno-deprecated-declarations
CFLAGS  ?= $(shell hipconfig --cpp_config) -Igcc_includes -Ilibgomp_includes \
    -Wno-deprecated-declarations -Wno-discarded-qualifiers
FCFLAGS ?=

SRC = plugin-hip.c
OBJ = $(SRC:.c=.o)

.PHONY: libgomp clean

all: libgomp-plugin-hip.so libgomp-plugin-hip.a

$(OBJ): %.o: %.c
	$(CC) $(CFLAGS) -fPIC -c $^

prepare:
	mkdir -p lib
	
lib: libgomp-plugin-hip.so libgomp-plugin-hip.a
	mv libgomp-plugin-* lib/
libgomp-plugin-hip.so: prepare $(OBJ)
	gcc -shared -o $@ $(OBJ)
libgomp-plugin-hip.a: prepare $(OBJ)
	ar rcs $@ $(OBJ)
	mv $@ lib/
clean:
	rm -rf *.o
