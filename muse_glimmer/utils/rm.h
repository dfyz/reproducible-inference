#pragma once

#include <fenv.h>

#define RETURN_WITH_RM(rm, call) { \
    _Pragma("STDC FENV_ACCESS ON") \
    int prev_mode = fegetround();  \
    fesetround(rm);                \
    float res = (call);            \
    fesetround(prev_mode);         \
    return res;                    \
}
