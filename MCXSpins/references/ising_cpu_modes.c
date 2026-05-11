#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define DIM 2
#define L 64
#define N (L*L)
#define SWEEPS_EQUI 1000
#define SWEEPS_GLOBAL 100000
#define SWEEPS_LOCAL 1
#define SWEEPS_EMPTY 1
#define TOT_SWEEPS (SWEEPS_GLOBAL*SWEEPS_LOCAL*SWEEPS_EMPTY)
#define BETA 0.44
#define A 1664525u
#define C 1013904223u
#define MULT 2.328306437080797e-10f
#define RAN(n) (n = A*n + C)

typedef int spin_t;

enum SimMode {
    MODE_TABLE_LCG = 0,
    MODE_TABLE_STD = 1,
    MODE_CONT_STD = 2,
};

enum RngMode {
    RNG_LCG = 0,
    RNG_MT = 1,
    RNG_XOSHIRO = 2,
};

/* ---------------- MT19937 (32-bit) ---------------- */
#define MT_N 624
#define MT_M 397
static uint32_t mt[MT_N];
static int mt_index = MT_N + 1;

static void mt_seed(uint32_t seed)
{
    mt[0] = seed;
    for (int i = 1; i < MT_N; ++i) {
        mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + (uint32_t)i;
    }
    mt_index = MT_N;
}

static void mt_twist(void)
{
    for (int i = 0; i < MT_N; ++i) {
        uint32_t y = (mt[i] & 0x80000000u) | (mt[(i + 1) % MT_N] & 0x7fffffffu);
        mt[i] = mt[(i + MT_M) % MT_N] ^ (y >> 1);
        if (y & 1u) mt[i] ^= 0x9908b0dfu;
    }
    mt_index = 0;
}

static inline uint32_t mt_next_u32(void)
{
    if (mt_index >= MT_N) mt_twist();
    uint32_t y = mt[mt_index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

/* ---------------- xoshiro128** ---------------- */
static uint32_t xo_s[4] = {1u, 2u, 3u, 4u};

static inline uint32_t rotl32(uint32_t x, int k)
{
    return (x << k) | (x >> (32 - k));
}

static uint32_t splitmix32(uint32_t *x)
{
    uint32_t z = (*x += 0x9e3779b9u);
    z = (z ^ (z >> 16)) * 0x85ebca6bu;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    return z ^ (z >> 16);
}

static void xoshiro_seed(uint32_t seed)
{
    uint32_t x = seed;
    for (int i = 0; i < 4; ++i) xo_s[i] = splitmix32(&x);
}

static inline uint32_t xoshiro_next_u32(void)
{
    uint32_t result = rotl32(xo_s[1] * 5u, 7) * 9u;
    uint32_t t = xo_s[1] << 9;

    xo_s[2] ^= xo_s[0];
    xo_s[3] ^= xo_s[1];
    xo_s[1] ^= xo_s[2];
    xo_s[0] ^= xo_s[3];

    xo_s[2] ^= t;
    xo_s[3] = rotl32(xo_s[3], 11);

    return result;
}

static inline float randu_lcg(uint32_t *ran)
{
    return MULT * (float)RAN(*ran);
}

static inline float randu_mt(void)
{
    return MULT * (float)mt_next_u32();
}

static inline float randu_xoshiro(void)
{
    return MULT * (float)xoshiro_next_u32();
}

static int cpu_energy(spin_t *s)
{
    int ie = 0;
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y)
            ie += s[L*y+x] *
                  (s[L*y+((x==0)?L-1:x-1)] + s[L*y+((x==L-1)?0:x+1)] +
                   s[L*((y==0)?L-1:y-1)+x] + s[L*((y==L-1)?0:y+1)+x]);
    return ie / 2;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [mode] [rng] [seed]\n"
            "  mode: table-lcg | table-std | cont-std | standard\n"
            "  rng : mt | xoshiro (only for table-std/cont-std)\n"
            "  seed: optional unsigned int (default 3145627)\n"
            "\n"
            "Examples:\n"
            "  %s                     # run standard (all modes)\n"
            "  %s 123                 # run standard with seed 123\n"
            "  %s table-lcg 123\n"
            "  %s table-std mt 123\n"
            "  %s cont-std xoshiro 123\n",
            prog, prog, prog, prog, prog,
            prog);
}

static int is_uint_arg(const char *s)
{
    if (!s || !*s) return 0;
    for (const char *p = s; *p; ++p) {
        if (*p < '0' || *p > '9') return 0;
    }
    return 1;
}

static const char *sim_mode_str(enum SimMode mode)
{
    return (mode == MODE_TABLE_LCG) ? "table-lcg" :
           (mode == MODE_TABLE_STD) ? "table-std" : "cont-std";
}

static const char *rng_mode_str(enum RngMode rng_mode)
{
    return (rng_mode == RNG_LCG) ? "lcg" :
           (rng_mode == RNG_MT) ? "mt" : "xoshiro";
}

static void print_header(void)
{
    printf("%-10s %-7s %8s %6s %10s %10s %10s %8s\n",
        "mode", "rng", "seed", "beta", "cpu_ms", "ns/flip", "updates", "final_E");
    printf("%-10s %-7s %8s %6s %10s %10s %10s %8s\n",
        "----------", "-------", "--------", "------", "----------", "----------", "----------", "--------");
}

static int simulate_mode(enum SimMode mode, enum RngMode rng_mode, uint32_t seed)
{
    spin_t *s = (spin_t*)malloc(N * sizeof(spin_t));
    if (!s) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    float boltz[2*DIM+1];
    for (int i = 0; i <= 2*DIM; ++i) boltz[i] = expf(-2.0f * (float)BETA * (float)i);

    uint32_t ran = seed;
    if (rng_mode == RNG_MT) mt_seed(seed);
    if (rng_mode == RNG_XOSHIRO) xoshiro_seed(seed);

    for (int i = 0; i < N; ++i) {
        float u;
        if (rng_mode == RNG_LCG) u = randu_lcg(&ran);
        else if (rng_mode == RNG_MT) u = randu_mt();
        else u = randu_xoshiro();
        s[i] = (u < 0.5f) ? 1 : -1;
    }

    for (int i = 0; i < SWEEPS_EQUI; ++i) {
        for (int j = 0; j < SWEEPS_EMPTY * SWEEPS_LOCAL; ++j) {
            for (int y = 0; y < L; ++y) {
                for (int x = 0; x < L; ++x) {
                    int ide = s[L*y+x] *
                              (s[L*y+((x==0)?L-1:x-1)] + s[L*y+((x==L-1)?0:x+1)] +
                               s[L*((y==0)?L-1:y-1)+x] + s[L*((y==L-1)?0:y+1)+x]);

                    float u;
                    if (rng_mode == RNG_LCG) u = randu_lcg(&ran);
                    else if (rng_mode == RNG_MT) u = randu_mt();
                    else u = randu_xoshiro();

                    if (mode == MODE_CONT_STD) {
                        int dE = 2 * ide;
                        if (dE <= 0 || u < expf(-(float)BETA * (float)dE)) s[L*y+x] = -s[L*y+x];
                    } else {
                        if (ide <= 0 || u < boltz[ide]) s[L*y+x] = -s[L*y+x];
                    }
                }
            }
        }
    }

    int ie = cpu_energy(s);
    uint64_t updates = 0;

    struct timespec starttime, stoptime;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &starttime);

    for (int i = 0; i < SWEEPS_GLOBAL; ++i) {
        for (int j = 0; j < SWEEPS_EMPTY * SWEEPS_LOCAL; ++j) {
            for (int y = 0; y < L; ++y) {
                for (int x = 0; x < L; ++x) {
                    ++updates;
                    int ide = s[L*y+x] *
                              (s[L*y+((x==0)?L-1:x-1)] + s[L*y+((x==L-1)?0:x+1)] +
                               s[L*((y==0)?L-1:y-1)+x] + s[L*((y==L-1)?0:y+1)+x]);

                    float u;
                    if (rng_mode == RNG_LCG) u = randu_lcg(&ran);
                    else if (rng_mode == RNG_MT) u = randu_mt();
                    else u = randu_xoshiro();

                    if (mode == MODE_CONT_STD) {
                        int dE = 2 * ide;
                        if (dE <= 0 || u < expf(-(float)BETA * (float)dE)) {
                            s[L*y+x] = -s[L*y+x];
                            ie -= dE;
                        }
                    } else {
                        if (ide <= 0 || u < boltz[ide]) {
                            s[L*y+x] = -s[L*y+x];
                            ie -= 2 * ide;
                        }
                    }
                }
            }
        }
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stoptime);

    long double elapsed = 1e9L * (stoptime.tv_sec - starttime.tv_sec) +
                          (stoptime.tv_nsec - starttime.tv_nsec);

        printf("%-10s %-7s %8u %6.2f %10.3Lf %10.6Lf %10llu %8d\n",
            sim_mode_str(mode), rng_mode_str(rng_mode), seed,
            (double)BETA,
            elapsed/1000000.0L, elapsed/((long double)N)/TOT_SWEEPS,
            (unsigned long long)updates, ie);

    free(s);
    return 0;
}

int main(int argc, char *argv[])
{
    enum SimMode mode;
    enum RngMode rng_mode = RNG_LCG;
    uint32_t seed = 3145627u;

    if (argc < 2) {
        print_header();
        if (simulate_mode(MODE_TABLE_LCG, RNG_LCG, seed) != 0) return 1;
        if (simulate_mode(MODE_TABLE_STD, RNG_MT, seed) != 0) return 1;
        if (simulate_mode(MODE_TABLE_STD, RNG_XOSHIRO, seed) != 0) return 1;
        if (simulate_mode(MODE_CONT_STD, RNG_MT, seed) != 0) return 1;
        if (simulate_mode(MODE_CONT_STD, RNG_XOSHIRO, seed) != 0) return 1;
        return 0;
    }

    if (strcmp(argv[1], "standard") == 0) {
        if (argc >= 3) seed = (uint32_t)strtoul(argv[2], NULL, 10);

        print_header();
        if (simulate_mode(MODE_TABLE_LCG, RNG_LCG, seed) != 0) return 1;
        if (simulate_mode(MODE_TABLE_STD, RNG_MT, seed) != 0) return 1;
        if (simulate_mode(MODE_TABLE_STD, RNG_XOSHIRO, seed) != 0) return 1;
        if (simulate_mode(MODE_CONT_STD, RNG_MT, seed) != 0) return 1;
        if (simulate_mode(MODE_CONT_STD, RNG_XOSHIRO, seed) != 0) return 1;
        return 0;
    }

    if (is_uint_arg(argv[1])) {
        seed = (uint32_t)strtoul(argv[1], NULL, 10);
        print_header();
        if (simulate_mode(MODE_TABLE_LCG, RNG_LCG, seed) != 0) return 1;
        if (simulate_mode(MODE_TABLE_STD, RNG_MT, seed) != 0) return 1;
        if (simulate_mode(MODE_TABLE_STD, RNG_XOSHIRO, seed) != 0) return 1;
        if (simulate_mode(MODE_CONT_STD, RNG_MT, seed) != 0) return 1;
        if (simulate_mode(MODE_CONT_STD, RNG_XOSHIRO, seed) != 0) return 1;
        return 0;
    }

    if (strcmp(argv[1], "table-lcg") == 0) {
        mode = MODE_TABLE_LCG;
        rng_mode = RNG_LCG;
    } else if (strcmp(argv[1], "table-std") == 0) {
        mode = MODE_TABLE_STD;
        rng_mode = RNG_MT;
    } else if (strcmp(argv[1], "cont-std") == 0) {
        mode = MODE_CONT_STD;
        rng_mode = RNG_MT;
    } else {
        usage(argv[0]);
        return 1;
    }

    if (mode != MODE_TABLE_LCG && argc >= 3) {
        if (strcmp(argv[2], "mt") == 0) rng_mode = RNG_MT;
        else if (strcmp(argv[2], "xoshiro") == 0) rng_mode = RNG_XOSHIRO;
        else {
            usage(argv[0]);
            return 1;
        }
    }

    if ((mode == MODE_TABLE_LCG && argc >= 3) || (mode != MODE_TABLE_LCG && argc >= 4)) {
        const char *seed_arg = (mode == MODE_TABLE_LCG) ? argv[2] : argv[3];
        seed = (uint32_t)strtoul(seed_arg, NULL, 10);
    }

    print_header();
    return simulate_mode(mode, rng_mode, seed);
}
