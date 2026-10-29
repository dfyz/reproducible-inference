// clang -O2 -std=c23 -fopenmp -march=native -lm -o check_mfu check_mfu.c
#include "utils/ex.h"
#include "utils/rcp.h"

#include "../ptx_math_exp2f.h"
#include "../ptx_math_recip.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

union fp32_int {
    float    f;
    uint32_t i;
};

typedef float (*mfu_func)(float);

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "specify the function to test: ex2|rcp\n");
        exit(2);
    }

    mfu_func ref_func, our_func;
    const char* func = argv[1];
    if (strcmp(func, "ex2") == 0) {
        ref_func = ptxm_ex2_sm5x;
        our_func = ex2;
    } else if (strcmp(func, "rcp") == 0) {
        ref_func = ptxm_rcp_sm5x;
        our_func = rcp;
    } else {
        fprintf(stderr, "unknown function %s\n", func);
        exit(3);
    }

    #pragma omp parallel for
    for (uint64_t ii = 0; ii <= UINT_MAX; ++ii) {
        if (ii % 100'000'000 == 0) {
            fprintf(stderr, "%lu/%u\n", ii, UINT_MAX + 1);
        }

        union fp32_int tmp = {.i = ii};

        if (!isnan(tmp.f)) {
            float ref = ref_func(tmp.f);
            float our = our_func(tmp.f);

            if (ref != our) {
                #pragma omp critical
                {
                    printf("x   = %.13a\n", tmp.f);
                    printf("ref = %.13a\n", ref);
                    printf("our = %.13a\n", our);
                }
                exit(1);
            }
        }
    }

    puts("ALL GOOD");
}
