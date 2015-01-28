# Actions

CC=gcc
LD=$(CC)
TARGET=a.out
OBJ_EXT=o
CLEAN=rm -f *.out *.exe *.obj *.lib *.o *.a *.s *.i *.I

all:		
	$(CM_PREFIX) $(CC) $(CFLAGS) -c *.c 
	$(LD) *.$(OBJ_EXT) $(LD_FLAGS) -lm -o $(TARGET)
clean:
	$(CLEAN)
