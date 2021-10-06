CC ?= gcc
FC ?= gfortran

#CFLAGS  ?= $(shell hipconfig --cpp_config) -I../ -I../../include \
#    -Wno-deprecated-declarations
CFLAGS  ?= $(shell hipconfig --cpp_config) -Igcc_includes -Ilibgomp_includes \
    -Wno-deprecated-declarations
FCFLAGS ?=

SRC = plugin-hip.c
OBJ = $(SRC:.c=.o)

.PHONY: libgomp clean

$(OBJ): %.o: %.c
	$(CC) $(CFLAGS) -c $^

prepare:
	mkdir -p lib
	
lib: libgomp-plugin-g.so libgomp-plugin-hipamd.a
	mv libgomp-plugin-* lib/
libgomp-plugin-g.so: prepare $(OBJ)
	gcc -shared -o $@ $(OBJ)
libgomp-plugin-g.a: prepare $(OBJ)
	ar rcs $@ $(OBJ)
	mv $@ lib/
clean:
	rm -rf *.o *.mod
