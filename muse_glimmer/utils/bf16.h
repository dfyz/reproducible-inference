#pragma once

#include <stdint.h>
#include <string.h>

typedef struct {
    uint16_t raw;
} bf16;

float to_float(bf16 x) {
    float res = 0.0f;
    memcpy((char*)&res + 2, &x, sizeof(x));
    return res;
}

bf16 to_bf16(float x) {
    uint32_t x_int;
    memcpy(&x_int, &x, sizeof(x));
    uint32_t bias = (x_int >> 16) & 1;
    x_int = x_int + 0x7f'ffu + bias;

    bf16 res;
    memcpy(&res, (char*)&x_int + 2, sizeof(res));
    return res;
}

float trunc_to_bf16(float x) {
    return to_float(to_bf16(x));
}