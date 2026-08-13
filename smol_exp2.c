// gcc -O2 -std=c23 -o smol_exp2 smol_exp2.c -march=native -lm
#include "ptx_math_exp2f.h"

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

double truncate(double x, int last_bit) {
    constexpr int FP64_MANTISSA_BITS = 52;
    double magic = 1ULL << (FP64_MANTISSA_BITS + last_bit);
    return x + magic - magic;
}

double approx_square(double x) {
    double x_orig = x;
    double res = 0.0f;
    for (double shift = 0x1p-7; shift >= 0x1p-23; shift *= 0x1p-1) {
        if (x >= shift) {
            res += truncate(x_orig * shift, -28);
            x -= shift;
        }
    }
    return truncate(res, -27);
}

float smol_exp2(float x) {
    double coefs[64][3] = {
        { 0x1.000001df88p+0, 0x1.62e4p-1, 0x1.ee0p-3 },
        { 0x1.02c9a55f88p+0, 0x1.66c0p-1, 0x1.f50p-3 },
        { 0x1.059b0f5f88p+0, 0x1.6aa8p-1, 0x1.fa0p-3 },
        { 0x1.087452df88p+0, 0x1.6e9cp-1, 0x1.ff0p-3 },
        { 0x1.0b55895f88p+0, 0x1.7298p-1, 0x1.030p-2 },
        { 0x1.0e3ec4df88p+0, 0x1.76a4p-1, 0x1.048p-2 },
        { 0x1.11301e5f88p+0, 0x1.7ab8p-1, 0x1.078p-2 },
        { 0x1.1429acdf88p+0, 0x1.7ed8p-1, 0x1.0a0p-2 },
        { 0x1.172b865f88p+0, 0x1.8300p-1, 0x1.0e8p-2 },
        { 0x1.1a35c0df88p+0, 0x1.8738p-1, 0x1.110p-2 },
        { 0x1.1d48745f88p+0, 0x1.8b7cp-1, 0x1.138p-2 },
        { 0x1.2063badf88p+0, 0x1.8fc8p-1, 0x1.178p-2 },
        { 0x1.2387a8df88p+0, 0x1.9424p-1, 0x1.1a0p-2 },
        { 0x1.26b4585f88p+0, 0x1.988cp-1, 0x1.1c0p-2 },
        { 0x1.29e9e1df88p+0, 0x1.9cfcp-1, 0x1.208p-2 },
        { 0x1.2d285cdf88p+0, 0x1.a17cp-1, 0x1.238p-2 },
        { 0x1.306fe2df88p+0, 0x1.a608p-1, 0x1.268p-2 },
        { 0x1.33c08d5f88p+0, 0x1.aaa0p-1, 0x1.2a0p-2 },
        { 0x1.371a74df88p+0, 0x1.af48p-1, 0x1.2c0p-2 },
        { 0x1.3a7db55f88p+0, 0x1.b3f8p-1, 0x1.308p-2 },
        { 0x1.3dea665f88p+0, 0x1.b8b8p-1, 0x1.338p-2 },
        { 0x1.4160a45f88p+0, 0x1.bd84p-1, 0x1.370p-2 },
        { 0x1.44e0875f88p+0, 0x1.c260p-1, 0x1.398p-2 },
        { 0x1.486a2cdf88p+0, 0x1.c748p-1, 0x1.3c8p-2 },
        { 0x1.4bfdaedf88p+0, 0x1.cc3cp-1, 0x1.408p-2 },
        { 0x1.4f9b28df88p+0, 0x1.d140p-1, 0x1.438p-2 },
        { 0x1.5342b6df88p+0, 0x1.d650p-1, 0x1.478p-2 },
        { 0x1.56f4755f88p+0, 0x1.db70p-1, 0x1.4a8p-2 },
        { 0x1.5ab07f5f88p+0, 0x1.e09cp-1, 0x1.4f0p-2 },
        { 0x1.5e76f2df88p+0, 0x1.e5d8p-1, 0x1.528p-2 },
        { 0x1.6247ed5f88p+0, 0x1.eb20p-1, 0x1.570p-2 },
        { 0x1.662389df88p+0, 0x1.f07cp-1, 0x1.598p-2 },
        { 0x1.6a09e7df88p+0, 0x1.f5e4p-1, 0x1.5d8p-2 },
        { 0x1.6dfb25df88p+0, 0x1.fb5cp-1, 0x1.608p-2 },
        { 0x1.71f760df88p+0, 0x1.0070p+0, 0x1.658p-2 },
        { 0x1.75feb6df88p+0, 0x1.033cp+0, 0x1.688p-2 },
        { 0x1.7a1148df88p+0, 0x1.060ep+0, 0x1.6d0p-2 },
        { 0x1.7e2f355f88p+0, 0x1.08e8p+0, 0x1.718p-2 },
        { 0x1.82589bdf88p+0, 0x1.0bcap+0, 0x1.760p-2 },
        { 0x1.868d9b5f88p+0, 0x1.0eb6p+0, 0x1.788p-2 },
        { 0x1.8ace55df88p+0, 0x1.11a8p+0, 0x1.7d8p-2 },
        { 0x1.8f1aeb5f88p+0, 0x1.14a2p+0, 0x1.828p-2 },
        { 0x1.93737d5f88p+0, 0x1.17a6p+0, 0x1.858p-2 },
        { 0x1.97d82bdf88p+0, 0x1.1ab2p+0, 0x1.898p-2 },
        { 0x1.9c4919df88p+0, 0x1.1dc6p+0, 0x1.8e0p-2 },
        { 0x1.a0c6695f88p+0, 0x1.20e2p+0, 0x1.930p-2 },
        { 0x1.a5503cdf88p+0, 0x1.2408p+0, 0x1.968p-2 },
        { 0x1.a9e6b6df88p+0, 0x1.2736p+0, 0x1.9b0p-2 },
        { 0x1.ae89fbdf88p+0, 0x1.2a6cp+0, 0x1.a08p-2 },
        { 0x1.b33a2d5f88p+0, 0x1.2dacp+0, 0x1.a50p-2 },
        { 0x1.b7f770df88p+0, 0x1.30f6p+0, 0x1.a88p-2 },
        { 0x1.bcc1eb5f88p+0, 0x1.3448p+0, 0x1.ad0p-2 },
        { 0x1.c199c05f88p+0, 0x1.37a2p+0, 0x1.b30p-2 },
        { 0x1.c67f155f88p+0, 0x1.3b08p+0, 0x1.b68p-2 },
        { 0x1.cb720fdf88p+0, 0x1.3e76p+0, 0x1.bb8p-2 },
        { 0x1.d072d6df88p+0, 0x1.41eep+0, 0x1.c00p-2 },
        { 0x1.d581905f88p+0, 0x1.456ep+0, 0x1.c68p-2 },
        { 0x1.da9e625f88p+0, 0x1.48fap+0, 0x1.ca8p-2 },
        { 0x1.dfc9755f88p+0, 0x1.4c90p+0, 0x1.ce8p-2 },
        { 0x1.e502f05f88p+0, 0x1.502ep+0, 0x1.d50p-2 },
        { 0x1.ea4afc5f88p+0, 0x1.53d8p+0, 0x1.d90p-2 },
        { 0x1.efa1c15f88p+0, 0x1.578ap+0, 0x1.df8p-2 },
        { 0x1.f50767df88p+0, 0x1.5b48p+0, 0x1.e48p-2 },
        { 0x1.fa7c1a5f88p+0, 0x1.5f10p+0, 0x1.ea0p-2 },
    };

    constexpr float n_bases = 0x1p6;

    float integral = truncf(x);
    float frac     = x - integral;
    float base_idx = truncf(frac*n_bases);

    double offset  = truncate(frac - base_idx/n_bases, -23);
    double* cc     = coefs[(size_t)base_idx];
    return ldexp(
        cc[0] + offset*cc[1] + approx_square(offset)*cc[2],
        (int)integral
    );
}

union fp32_int {
    float    f;
    uint32_t i;
};

int main() {
    int orig_rounding = fegetround();
    for (uint64_t ii = 0; ii <= UINT_MAX; ++ii) {
        union fp32_int tmp = {.i = ii};
        // Only non-negative numbers are handled.
        if (!isfinite(tmp.f) || tmp.f < 0.0f) {
            continue;
        }

        float ref = ptxm_ex2_sm5x(tmp.f);
        if (!isfinite(ref)) {
            continue;
        }


        fesetround(FE_TOWARDZERO);
        float our = smol_exp2(tmp.f);
        fesetround(orig_rounding);

        if (ii % 100'000'000 == 0) {
            fprintf(stderr, "%lu/%lu\n", ii, UINT_MAX);
        }

        if (ref != our) {
            printf("x   = %.13a\n", tmp.f);
            printf("ref = %.13a\n", ref);
            printf("our = %.13a\n", our);
            exit(1);
        }
    }
}
