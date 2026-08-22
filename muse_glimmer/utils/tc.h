#pragma once

#include "bf16.h"
#include "minmax.h"

#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

constexpr size_t K_DIM = 16;

int32_t get_exp(float x) {
    uint32_t x_int;
    memcpy(&x_int, &x, sizeof(x));
    int32_t b_exp = (x_int >> 23) & 0xff;
    return (b_exp + (b_exp == 0)) - 127;
}

double shift(double x, double magic) {
    magic = copysign(magic, x);
    return x + magic - magic;
}

int32_t guard_zero(double res, int32_t exp) {
    return res == 0.0 ? -133 : exp;
}

float load_bf16(const bf16* x, size_t ii, size_t n_elems) {
    return ii < n_elems ? to_float(x[ii]) : 0.0f;
}

float __tc_bf16_fp32(float c, const bf16* a, const bf16* b, size_t n_elems) {
    double addends[K_DIM];

    for (size_t off = 0; off < n_elems; off += K_DIM) {
        int32_t max_exp = guard_zero(c, get_exp(c));

        for (size_t ii = off; ii < off + K_DIM; ++ii) {
            float lhs = load_bf16(a, ii, n_elems);
            float rhs = load_bf16(b, ii, n_elems);

            double prod = (double)lhs * (double)rhs;
            int32_t cur_exp = guard_zero(prod, get_exp(lhs) + get_exp(rhs));
            max_exp = maxf(max_exp, cur_exp);
            addends[ii - off] = prod;
        }

        double magic = ldexp(0x1p27, max_exp);
        double acc = shift(c, magic);
        for (size_t ii = 0; ii < K_DIM; ++ii) {
            acc += shift(addends[ii], magic);
        }
        c = acc;
    }

    return c;
}

float tc_bf16_fp32(float c, const bf16* a, const bf16* b, size_t n_elems) {
    int prev_mode = fegetround();
    fesetround(FE_TOWARDZERO);
    float res = __tc_bf16_fp32(c, a, b, n_elems);
    fesetround(prev_mode);
    return res;
}
