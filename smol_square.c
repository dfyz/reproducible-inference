// gcc -O2 -std=c23 -o smol_square smol_square.c -march=native -lm
#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#define MASK_U64(numbits) ((UINT64_C(1) << (numbits)) - 1u)

double int_square_exact(uint64_t x) {
    return ldexp(x * x, -46);
}

double int_square(uint64_t x) {
    uint64_t res = 0;
    for (size_t ii = 2; ii < 17; ++ii) {
        if (x & (1ull << ii)) {
            res += (x << ii) & ~MASK_U64(18);
        }
    }
    res &= ~MASK_U64(19);

    return ldexp(res, -46);
}

double float_square_exact(uint64_t x) {
    double y = ldexp(x, -23);
    return y * y;
}

double float_square(uint64_t x) {
    fesetround(FE_TOWARDZERO);
    double y = ldexp(x, -23);
    double z = y;

    double res = 0.0f;
    for (double shift = 0x1p-7; shift >= 0x1p-23; shift *= 0x1p-1) {
        if (z >= shift) {
            res += y * shift + 0x1p24 - 0x1p24;
            z -= shift;
        }
    }
    res = res + 0x1p25 - 0x1p25;
    fesetround(FE_TONEAREST);
    return res;
}

int main() {
    for (uint64_t ii = 0; ii < 131072; ++ii) {
        double sq_int_exact   = int_square_exact(ii);
        double sq_int         = int_square(ii);
        double sq_float_exact = float_square_exact(ii);
        double sq_float       = float_square(ii);

        puts("===");
        printf("x = %lu\n", ii);
        printf("I = %.13a\n", sq_int_exact);
        printf("F = %.13a\n", sq_float_exact);
        printf("i = %.13a\n", sq_int);
        printf("f = %.13a\n", sq_float);

        if (sq_int_exact != sq_float_exact || sq_int != sq_float) {
            return 1;
        }
    }
}