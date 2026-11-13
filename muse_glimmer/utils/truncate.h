#pragma once

#include <math.h>
#include <stdint.h>

// Drop every place of `x` below 2^-threshold.
// Assumes FE_TOWARDZERO is active.
// E.g., truncate(2^-3 + 2^-6 + 2^-10, 9)
//              = 2^-3 + 2^-6
double truncate(double x, int threshold) {
    // For this trick to work, we assume that `|x|`
    // is comfortably smaller than `magic`.
    union flint {
        double   f;
        uint64_t i;
    };
    union flint tmp = {.f = x};
    union flint magic = {
             // the exponent is biased `52 - threshold`
        .i = ((1075ull - threshold) << 52) |
             // the sign is copied from `x`
             (tmp.i & (1ull << 63)),
    };
    return x + magic.f - magic.f;
}
