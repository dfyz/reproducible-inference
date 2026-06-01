#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

inline uint32_t float_as_u32(float x)
{
    uint32_t r;
    memcpy(&r, &x, 4u);

    return r;
}

inline float u32_as_float(uint32_t x)
{
    float r;
    memcpy(&r, &x, 4u);

    return r;
}

float fexp2i(int n)
{
    if (n > 127) return INFINITY;
    if (n < -126) return 0.0f;

    return ldexpf(1.0f, n);
}

#define PTX_CANONICAL_NAN UINT32_C(0x7fffffff)

inline float ptxm_nan(void)
{
    return u32_as_float(PTX_CANONICAL_NAN);
}

#define EXTRACT_BITS(x, count, from) \
    (((x) >> ((from) - (count))) & MASK_U32(count))

#define UPPER_SIGNIFICAND(x, m) \
    EXTRACT_BITS(x, m, 23)

#define LOWER_SIGNIFICAND(x, m) \
    ((x) & MASK_U32(23 - (m)))

#define FP_FORMAT(sign, ex, frac) \
    (((sign) << 31) | ((ex) << 23) | (frac))

#define MASK_U32(numbits) ((UINT32_C(1) << (numbits)) - 1u)

typedef struct ptxm_params
{
    const uint32_t (*table)[3];
    uint64_t bias;
} ptxm_params;


#define TRUNC_COLS 19
#define PPM_ROWS 9

static uint32_t ppm_antidiagonal(uint32_t x)
{
    return _pdep_u32(x, UINT32_C(0x55555555));
}

static uint32_t ppm_row(uint32_t x, int i)
{
    if (!((x >> i) & 1u))
        return 0;

    x <<= i + 1;
    x &= ~MASK_U32(2 * i + 2);

    return x;
}

uint64_t ptxm_square_approx(uint32_t x)
{
    uint64_t error = ppm_antidiagonal(x) & MASK_U32(TRUNC_COLS);

    for (int i = 0; i < PPM_ROWS; i++)
    {
        error += ppm_row(x, i) & MASK_U32(TRUNC_COLS);
    }

    const uint64_t exact = (uint64_t)x * x;

    return exact - error;
}
