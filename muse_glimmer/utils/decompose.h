#pragma once

#include <math.h>

void decompose(float x, float* frac, int* exp) {
    // Convert PDP-11-style fractions/exponents to IEEE-754-style.
    *frac = frexpf(x, exp);
    *frac *= 2.0f;
    --*exp;
}
