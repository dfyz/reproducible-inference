#pragma once

#include <fenv.h>

float with_rm(int rm, float(*f)(float), float x) {
    int prev_mode = fegetround();
    fesetround(rm);
    float res = f(x);
    fesetround(prev_mode);
    return res;
}
