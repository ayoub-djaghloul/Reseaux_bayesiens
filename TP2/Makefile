
CC=g++
CFLAGS=-g
LDFLAGS=
EXEC=use_hmm

all: $(EXEC)

use_hmm: use_hmm.o model.o
	$(CC) -o $@ $^ $(LDFLAGS)

model.o: hmm/model.cpp hmm/model.h
	$(CC) -o $@ -c $< $(CFLAGS)

use_hmm.o: hmm/use_hmm.cpp hmm/use_hmm.h
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf *.o

mrproper: clean
	rm -rf $(EXEC)
