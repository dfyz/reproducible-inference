#pragma once

#include "bf16.h"
#include "minmax.h"
#include "truncate.h"

#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

constexpr size_t K_DIM = 16;

int32_t get_exp(float x) {
    return maxi(-126, ilogbf(x));
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

        int threshold = 25 - max_exp;
        double acc = truncate(c, threshold);
        for (size_t ii = 0; ii < K_DIM; ++ii) {
            acc += truncate(addends[ii], threshold);
        }
        c = acc;
    }

    return c;
}

float tc_bf16_fp32(float c, const bf16* a, const bf16* b, size_t n_elems) {
    RETURN_WITH_RM(
        FE_TOWARDZERO,
        __tc_bf16_fp32(c, a, b, n_elems)
    );
}
