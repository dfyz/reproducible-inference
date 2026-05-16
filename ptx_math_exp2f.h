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

#define EX2_M 6

#define EX2_C0_TERM_ALIGNMENT 32
#define EX2_C1_TERM_ALIGNMENT 19
#define EX2_C2_TERM_ALIGNMENT 0

#define EX2_SUM_WEIGHT 57

typedef struct ptxm_params
{
    const uint32_t (*table)[3];
    uint64_t bias;
} ptxm_params;

const uint32_t ptxm_ex2_table[64][3] =
{
    {0x2000002, 0x58b9, 0x1ee}, {0x2059349, 0x59b0, 0x1f5},
    {0x20b361d, 0x5aaa, 0x1fa}, {0x210e8a4, 0x5ba7, 0x1ff},
    {0x216ab11, 0x5ca6, 0x206}, {0x21c7d88, 0x5da9, 0x209},
    {0x222603b, 0x5eae, 0x20f}, {0x2285358, 0x5fb6, 0x214},
    {0x22e570b, 0x60c0, 0x21d}, {0x2346b80, 0x61ce, 0x222},
    {0x23a90e7, 0x62df, 0x227}, {0x240c774, 0x63f2, 0x22f},
    {0x2470f50, 0x6509, 0x234}, {0x24d68af, 0x6623, 0x238},
    {0x253d3c2, 0x673f, 0x241}, {0x25a50b8, 0x685f, 0x247},
    {0x260dfc4, 0x6982, 0x24d}, {0x2678119, 0x6aa8, 0x254},
    {0x26e34e8, 0x6bd2, 0x258}, {0x274fb69, 0x6cfe, 0x261},
    {0x27bd4cb, 0x6e2e, 0x267}, {0x282c147, 0x6f61, 0x26e},
    {0x289c10d, 0x7098, 0x273}, {0x290d458, 0x71d2, 0x279},
    {0x297fb5c, 0x730f, 0x281}, {0x29f3650, 0x7450, 0x287},
    {0x2a6856c, 0x7594, 0x28f}, {0x2ade8e9, 0x76dc, 0x295},
    {0x2b560fd, 0x7827, 0x29e}, {0x2bcede4, 0x7976, 0x2a5},
    {0x2c48fd9, 0x7ac8, 0x2ae}, {0x2cc4712, 0x7c1f, 0x2b3},
    {0x2d413ce, 0x7d79, 0x2bb}, {0x2dbf64a, 0x7ed7, 0x2c1},
    {0x2e3eec0, 0x8038, 0x2cb}, {0x2ebfd6c, 0x819e, 0x2d1},
    {0x2f42290, 0x8307, 0x2da}, {0x2fc5e69, 0x8474, 0x2e3},
    {0x304b136, 0x85e5, 0x2ec}, {0x30d1b35, 0x875b, 0x2f1},
    {0x3159caa, 0x88d4, 0x2fb}, {0x31e35d5, 0x8a51, 0x305},
    {0x326e6f9, 0x8bd3, 0x30b}, {0x32fb056, 0x8d59, 0x313},
    {0x3389232, 0x8ee3, 0x31c}, {0x3418cd1, 0x9071, 0x326},
    {0x34aa078, 0x9204, 0x32d}, {0x353cd6c, 0x939b, 0x336},
    {0x35d13f6, 0x9536, 0x341}, {0x3667459, 0x96d6, 0x34a},
    {0x36feee0, 0x987b, 0x351}, {0x37983d5, 0x9a24, 0x35a},
    {0x383337f, 0x9bd1, 0x366}, {0x38cfe29, 0x9d84, 0x36d},
    {0x396e41e, 0x9f3b, 0x377}, {0x3a0e5ac, 0xa0f7, 0x380},
    {0x3ab031f, 0xa2b7, 0x38d}, {0x3b53cc3, 0xa47d, 0x395},
    {0x3bf92e9, 0xa648, 0x39d}, {0x3ca05df, 0xa817, 0x3aa},
    {0x3d495f7, 0xa9ec, 0x3b2}, {0x3df4381, 0xabc5, 0x3bf},
    {0x3ea0ece, 0xada4, 0x3c9}, {0x3f4f833, 0xaf88, 0x3d4}
};


static const ptxm_params model_params =
{
    .table = ptxm_ex2_table,
    .bias = UINT64_C(0x6fc4000000000000)
};

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

uint32_t ptxm_rro_ex2_sm5x(float x)
{
    const uint32_t sign = signbit(x) ? 1u : 0u;

    if (isnan(x)) return FP_FORMAT(sign, 0x80u, 0u);

    if (x >= 128.0f) return FP_FORMAT(0u, 0x81u, 0u);
    if (x <= -128.0f) return FP_FORMAT(1u, 0x81u, 0u);

    float integral;
    x = modff(x, &integral);

    uint32_t x_bits = float_as_u32(x);

    if (x != 0.0f)
    {
        x_bits |= UINT32_C(1) << 23;
        x_bits &= MASK_U32(24);

        x_bits >>= min(-ilogbf(x), 24);
    }

    const uint32_t i = fabsf(integral);
    const uint32_t r = x_bits & MASK_U32(23);

    return FP_FORMAT(sign, i, r);
}

float mufu_ex2(uint32_t reduced, const ptxm_params *params)
{
    uint32_t sign = reduced >> 31;
    int32_t integral = (reduced >> 23) & MASK_U32(8);

    switch (integral & 0x81u)
    {
    case 0x80u:
        return ptxm_nan();
    case 0x81u:
        return sign ? 0.0f : INFINITY;
    }

    if (sign)
    {
        integral = -integral;

        if (reduced & MASK_U32(23))
        {
            reduced = ~reduced;
            integral -= 1;
        }
    }

    const uint32_t xh = UPPER_SIGNIFICAND(reduced, EX2_M);
    const uint32_t xl = LOWER_SIGNIFICAND(reduced, EX2_M);

    const uint32_t *const c = params->table[xh];

    uint64_t c0_term = c[0];
    uint64_t c1_term = c[1] * (uint64_t)xl;
    uint64_t c2_term = c[2] * ptxm_square_approx(xl);

    c0_term <<= EX2_C0_TERM_ALIGNMENT;
    c1_term <<= EX2_C1_TERM_ALIGNMENT;
    c2_term <<= EX2_C2_TERM_ALIGNMENT;

    uint64_t sum = c0_term + c1_term + c2_term;

    sum += params->bias >> ((64 - EX2_SUM_WEIGHT) + 23);

    const uint32_t r_frac = EXTRACT_BITS(sum, 23, EX2_SUM_WEIGHT);
    const uint32_t r_bits = FP_FORMAT(0u, 127u, r_frac);

    const float r = u32_as_float(r_bits);

    return fexp2i(integral) * r;
}

float ptxm_ex2_sm5x(float x)
{
    if (isnan(x)) return ptxm_nan();

    const float r = mufu_ex2(ptxm_rro_ex2_sm5x(x), &model_params);

    return r;
}
