#pragma once

#include "ex.h"
#include "../../ptx_math_recip.h"

float sigmoid_fast(float x) {
    return ptxm_rcp_sm5x(1.0f + ex_fast(-x));
}
