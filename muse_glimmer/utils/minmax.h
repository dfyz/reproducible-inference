#pragma once

#include <stdint.h>

size_t mins(size_t x, size_t y) {
    return x < y ? x : y;
}

int32_t maxi(int32_t x, int32_t y) {
    return x > y ? x : y;
}

float maxf(float x, float y) {
    return x > y ? x : y;
}
