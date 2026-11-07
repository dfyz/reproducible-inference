#pragma once

#include "ex.h"
#include "rcp.h"

float sigmoid_fast(float x) {
    return rcp(1.0f + ex_fast(-x));
}
