// gcc -O2 -o check_swiglu check_swiglu.c -lm -march=x86-64-v3
#include "core_math_expf.h"
#include "ptx_expf.h"
#include "ptx_math_recip.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <immintrin.h>

#define N_ELEMS (8192U * 512)
typedef unsigned short bf16;

bf16 SWISH_INPUT[N_ELEMS];
bf16 GATE_INPUT[N_ELEMS];
bf16 AFTER_SWIGLU[N_ELEMS];

void read(bf16* tensor, const char* file_name) {
    FILE* f = fopen(file_name, "rb");
    if (f == NULL) {
        fprintf(stderr, "Failed to open %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    if (fread(tensor, sizeof(bf16), N_ELEMS, f) != N_ELEMS) {
        fprintf(stderr, "Failed to read from %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    fclose(f);
}

float load(const bf16* tensor, size_t idx) {
    unsigned res_int = ((unsigned)tensor[idx]) << 16;
    float res;
    memcpy(&res, &res_int, sizeof(float));
    return res;
}

float to_bf16(float x) {
    unsigned x_int;
    memcpy(&x_int, &x, sizeof(unsigned));

    unsigned bias = (x_int >> 16) & 1;
    x_int += 0x7FFFU + bias;
    x_int &= 0xFFFF0000U;
    memcpy(&x, &x_int, sizeof(float));

    return x;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: SWISH_INPUT GATE_INPUT AFTER_SWIGLU\n");
        exit(EXIT_FAILURE);
    }

    read(SWISH_INPUT, argv[1]);
    read(GATE_INPUT, argv[2]);
    read(AFTER_SWIGLU, argv[3]);

    uint32_t exp_ptx_core_mismatches = 0;
    uint32_t exp_glibc_core_mismatches = 0;
    uint32_t exp_ptx_glibc_mismatches = 0;

    uint32_t recip_core_mismatches = 0;

    for (size_t ii = 0; ii < N_ELEMS; ++ii) {
        float ref = load(AFTER_SWIGLU, ii);

        float si = load(SWISH_INPUT, ii);
        float gi = load(GATE_INPUT, ii);

        float to_exp = -si;
        float core_exp = cr_expf(to_exp);
        float ptx_exp = ptx_expf(to_exp);
        float glibc_exp = expf(to_exp);

        if (ptx_exp != core_exp) {
            ++exp_ptx_core_mismatches;
        }

        if (glibc_exp != core_exp) {
            ++exp_glibc_core_mismatches;
        }

        if (ptx_exp != glibc_exp) {
            ++exp_ptx_glibc_mismatches;
        }

        float denom = 1.0f + ptx_exp;

        float ptx_div = si * ptxm_rcp_sm5x(denom);
        float core_div = si / denom;

        if (ptx_div != core_div) {
            ++recip_core_mismatches;
        }

        float ours_fp32 = ptx_div * gi;
        float ours_bf16 = to_bf16(ours_fp32);

        if (ours_bf16 != ref) {
            printf("ref = %e (%a), ours = %e (%a)\n", ref, ref, ours_bf16, ours_bf16);
        }
    }

    printf("Hardware exp wasn't correctly rounded %u/%u times\n", exp_ptx_core_mismatches, N_ELEMS);
    printf("Glibc exp wasn't correctly rounded %u/%u times\n", exp_glibc_core_mismatches, N_ELEMS);
    printf("Hardware exp didn't match glibc %u/%u times\n", exp_ptx_glibc_mismatches, N_ELEMS);
    printf("Hardware div wasn't correctly rounded %u/%u times\n", recip_core_mismatches, N_ELEMS);

    return 0; 
}
