#pragma once

#include <fenv.h>

#define RETURN_WITH_RM(rm, call) { \
    int prev_mode = fegetround();  \
    fesetround(rm);                \
    float res = (call);            \
    fesetround(prev_mode);         \
    return res;                    \
}
