#pragma once

#include <math.h>

// Drop every place of `x` below 2^-threshold.
// E.g., truncate(2^-3 + 2^-6 + 2^-10, 9)
//              = 2^-3 + 2^-6
double truncate(double x, int threshold) {
    // For this trick to work, we assume that `|x|`
    // is comfortably smaller than `magic`.
    double magic = copysign(ldexp(1.0, 52 - threshold), x);
    return x + magic - magic;
}
