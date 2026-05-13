CC ?= cc
CXX ?= c++
CFLAGS ?= -O3 -march=native -std=c11 -Wall -Wextra
CXXFLAGS ?= -O3 -march=native

REF := ising_cpu
MODES := ising_cpu_modes

.PHONY: all clean run-ref run-standard run-table-lcg run-table-mt run-table-xoshiro run-cont-mt run-cont-xoshiro

all: $(REF) $(MODES)

$(REF): ising_cpu.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

$(MODES): ising_cpu_modes.c
	$(CC) $(CFLAGS) -o $@ $< -lm

run-ref: $(REF)
	./$(REF) ref_

run-standard: $(MODES)
	./$(MODES) standard 3145627

run-table-lcg: $(MODES)
	./$(MODES) table-lcg 3145627

run-table-mt: $(MODES)
	./$(MODES) table-std mt 3145627

run-table-xoshiro: $(MODES)
	./$(MODES) table-std xoshiro 3145627

run-cont-mt: $(MODES)
	./$(MODES) cont-std mt 3145627

run-cont-xoshiro: $(MODES)
	./$(MODES) cont-std xoshiro 3145627

clean:
	rm -f $(REF) $(MODES)
