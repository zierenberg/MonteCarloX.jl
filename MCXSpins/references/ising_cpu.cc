#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define DIM 2
#define L 64
#define BLOCKL 16
#define GRIDL (L/BLOCKL)
#define BLOCKS ((GRIDL*GRIDL)/2)
#define THREADS ((BLOCKL*BLOCKL)/2)
#define N (L*L)
#define SPINS_PER_BLOCK (N/2)
#define TOTTHREADS (BLOCKS*THREADS)
#define SWEEPS_EQUI 10000
#define SWEEPS_GLOBAL 1000000
#define SWEEPS_LOCAL 1 
#define SWEEPS_EMPTY 1
#define TOT_SWEEPS (SWEEPS_GLOBAL*SWEEPS_LOCAL*SWEEPS_EMPTY)
//#define BETA 0.440686793509771
#define BETA 0.44
//#define BETA 0.25
#define A       1664525
#define C       1013904223
#define sS(x,y) sS[(y+1)*(BLOCKL+2)+x+1]
#define RAN(n) (n = A*n + C)
#define MULT 2.328306437080797e-10f
//#define MEAS

typedef int spin_t;

int cpu_energy(spin_t *s)
{
  int ie = 0;
  for(int x = 0; x < L; ++x)
    for(int y = 0; y < L; ++y)
      ie += s[L*y+x]*(s[L*y+((x==0)?L-1:x-1)]+s[L*y+((x==L-1)?0:x+1)]+s[L*((y==0)?L-1:y-1)+x]+s[L*((y==L-1)?0:y+1)+x]);
    

  return ie/2;
}

void simulate(char *file)
{
  spin_t *s;
  int ie;
  float boltz[2*DIM+1];
  char str[100];
  strcpy(str,file);
  FILE *fp;

  srand48(3145627);
  s = (spin_t*)malloc(N*sizeof(spin_t));
  for(int i = 0; i < N; ++i) s[i] = (drand48() < 0.5) ? 1 : -1;
  for(int i = 0; i <= 2*DIM; ++i) boltz[i] = exp(-2*BETA*i);

  int ran = 1;

  strcpy(str,file);
  strcat(str,"em.series%0");
  fp = fopen(str,"w");

  for(int i = 0; i < SWEEPS_EQUI; ++i) {
    for(int j = 0; j < SWEEPS_EMPTY*SWEEPS_LOCAL; ++j) {
      for(int y = 0; y < L; ++y) {
	for(int x = 0; x < L; ++x) {
	  int ide = s[L*y+x]*(s[L*y+((x==0)?L-1:x-1)]+s[L*y+((x==L-1)?0:x+1)]+s[L*((y==0)?L-1:y-1)+x]+s[L*((y==L-1)?0:y+1)+x]);
	  //if(L*y+x == 1) fprintf(stdout,"%llu\n",ran);
	  //RAN(ran);
	  if(ide <= 0 || MULT*(*(unsigned int*)(&RAN(ran))) < boltz[ide]) s[L*y+x] = -s[L*y+x];
	}
      }
    }
  }
  
  ie = cpu_energy(s);
  unsigned long long updates = 0ull;

  struct timespec starttime, stoptime;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &starttime);

  for(int i = 0; i < SWEEPS_GLOBAL; ++i) {
    for(int j = 0; j < SWEEPS_EMPTY*SWEEPS_LOCAL; ++j) {
      for(int y = 0; y < L; ++y) {
	for(int x = 0; x < L; ++x) {
    ++updates;
	  int ide = s[L*y+x]*(s[L*y+((x==0)?L-1:x-1)]+s[L*y+((x==L-1)?0:x+1)]+s[L*((y==0)?L-1:y-1)+x]+s[L*((y==L-1)?0:y+1)+x]);
	  //RAN(ran);
	  if(ide <= 0 || MULT*(*(unsigned int*)(&RAN(ran))) < boltz[ide]) {
	    s[L*y+x] = -s[L*y+x];
	    ie -= 2*ide;
	  }
	}
      }
    }
#ifdef MEAS
    fwrite(&ie, sizeof(int), 1, fp);
#endif
  }

#ifdef MEAS
  fclose(fp);
#endif

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stoptime);

  long double elapsed = 1e9* (stoptime.tv_sec - starttime.tv_sec) + (stoptime.tv_nsec - starttime.tv_nsec);
      printf("beta=%0.2f, CPU time: %Lf ms, %Lf ns per spin flip, updates=%llu\n",
        (double)BETA,
      elapsed/1000000,
         elapsed/((long double)N)/TOT_SWEEPS,
      updates);

  free(s);
}

int main(int argc, char *argv[])
{
  simulate(argv[1]);
}
