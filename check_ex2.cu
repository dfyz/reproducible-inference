#include <cstdlib>
#include <cstdio>
#include <cstdint>
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

    /*bool square_result = false;

    if (x < -126.0f)
    {
        x *= 0.5f;
        square_result = true;
    }*/

    const float y = ptxm_rro_ex2_sm5x(x);
    if (fpclassify(x) == FP_NORMAL && y != 0.0f) {
        printf("%g (%a) -> %g (%a)\n", x, x, y, y);
    }
    const float r = mufu_ex2(y, &model_params);

    return r;
    // return square_result ? r * r : r;
}

float SASS_RRO_EX2(float x)
{
    return u32_as_float(ptxm_rro_ex2_sm5x(x));
}

float SASS_MUFU_EX2(float x)
{
    return mufu_ex2(float_as_u32(x), &model_params);
}

#pragma STDC FENV_ACCESS ON

typedef union {float f; uint32_t u;} b32u32_u;
typedef union {double f; uint64_t u;} b64u64_u;

// deal with x=nan, x < -149 and x >= 128
static float as_special(float x){
  b32u32_u t = {.f = x};
  uint32_t ux = t.u<<1;
  if(ux >= 0xffu<<24) { // x is inf or nan
    if(ux > 0xffu<<24) return x + x; // x = nan
    static const float ir[] = {__builtin_inff(), 0.0f};
    return ir[t.u>>31]; // x = +-inf
  }
  if(t.u>=0xc3150000u){ // x < -149
    double z = x, y = 0x1p-149 + (z + 149)*0x1p-150;
    return __builtin_fmax(y, 0x1p-151);
  }
  // now x >= 128
  return 0x1p127f * 0x1p127f;
}

float cr_exp2f(float x){
  static const b64u64_u tb[] =
    {{0x1.0000000000000p+0}, {0x1.02c9a3e778061p+0}, {0x1.059b0d3158574p+0}, {0x1.0874518759bc8p+0},
     {0x1.0b5586cf9890fp+0}, {0x1.0e3ec32d3d1a2p+0}, {0x1.11301d0125b51p+0}, {0x1.1429aaea92de0p+0},
     {0x1.172b83c7d517bp+0}, {0x1.1a35beb6fcb75p+0}, {0x1.1d4873168b9aap+0}, {0x1.2063b88628cd6p+0},
     {0x1.2387a6e756238p+0}, {0x1.26b4565e27cddp+0}, {0x1.29e9df51fdee1p+0}, {0x1.2d285a6e4030bp+0},
     {0x1.306fe0a31b715p+0}, {0x1.33c08b26416ffp+0}, {0x1.371a7373aa9cbp+0}, {0x1.3a7db34e59ff7p+0},
     {0x1.3dea64c123422p+0}, {0x1.4160a21f72e2ap+0}, {0x1.44e086061892dp+0}, {0x1.486a2b5c13cd0p+0},
     {0x1.4bfdad5362a27p+0}, {0x1.4f9b2769d2ca7p+0}, {0x1.5342b569d4f82p+0}, {0x1.56f4736b527dap+0},
     {0x1.5ab07dd485429p+0}, {0x1.5e76f15ad2148p+0}, {0x1.6247eb03a5585p+0}, {0x1.6623882552225p+0},
     {0x1.6a09e667f3bcdp+0}, {0x1.6dfb23c651a2fp+0}, {0x1.71f75e8ec5f74p+0}, {0x1.75feb564267c9p+0},
     {0x1.7a11473eb0187p+0}, {0x1.7e2f336cf4e62p+0}, {0x1.82589994cce13p+0}, {0x1.868d99b4492edp+0},
     {0x1.8ace5422aa0dbp+0}, {0x1.8f1ae99157736p+0}, {0x1.93737b0cdc5e5p+0}, {0x1.97d829fde4e50p+0},
     {0x1.9c49182a3f090p+0}, {0x1.a0c667b5de565p+0}, {0x1.a5503b23e255dp+0}, {0x1.a9e6b5579fdbfp+0},
     {0x1.ae89f995ad3adp+0}, {0x1.b33a2b84f15fbp+0}, {0x1.b7f76f2fb5e47p+0}, {0x1.bcc1e904bc1d2p+0},
     {0x1.c199bdd85529cp+0}, {0x1.c67f12e57d14bp+0}, {0x1.cb720dcef9069p+0}, {0x1.d072d4a07897cp+0},
     {0x1.d5818dcfba487p+0}, {0x1.da9e603db3285p+0}, {0x1.dfc97337b9b5fp+0}, {0x1.e502ee78b3ff6p+0},
     {0x1.ea4afa2a490dap+0}, {0x1.efa1bee615a27p+0}, {0x1.f50765b6e4540p+0}, {0x1.fa7c1819e90d8p+0}};

  b32u32_u t = {.f = x};
  if(__builtin_expect((t.u&0xffff)==0, 0)){ // x maybe integer
    int k = ((t.u>>23)&0xff)-127; // 2^k <= |x| < 2^(k+1)
    if(__builtin_expect(k>=0 && k<9 && (t.u<<(9+k)) == 0, 0)){
      // x integer, with 1 <= |x| < 2^9
      int msk = (int)t.u>>31;
      int m = ((t.u&0x7fffff)|(1<<23))>>(23-k);
      m = (m^msk) - msk + 127;
      if(m>0 && m<255){
    t.u = m<<23;
    return t.f;
      } else if(m<=0 && m>-23){
        /* If f(x) underflows but is exact, no underflow exception should be
           raised (cf IEEE 754-2019). */
    t.u = 1<<(22+m);
        return t.f;
      }
    }
  }
  uint32_t ux = t.u<<1;
  if (__builtin_expect(ux>=0x86000000u || ux<0x65000000u, 0)){
    // |x| >= 128 or x=nan or |x| < 0x1p-26
    if(__builtin_expect(ux<0x65000000u, 1)) return 1.0f + x; // |x| < 0x1p-26
    // if x < -149 or 128 <= x we call as_special()
    if(!(t.u>=0xc3000000u && t.u<0xc3150000u)) return as_special(x);
  }
  double offd = 0x1.8p46, xd = x, h = xd - ((xd + offd) - offd), h2 = h*h;
  b32u32_u u = {.f = x + 0x1.8p17f};
  b64u64_u sv = tb[u.u&0x3f];
  sv.u += (uint64_t)(u.u>>6)<<52;
  static const double b[] = {1, 0x1.62e42fef4c4e7p-1, 0x1.ebfd1b232f475p-3, 0x1.c6b19384ecd93p-5};
  double r = sv.f*((b[0] + h*b[1]) + h2*(b[2] + h*b[3])), eps = 0x1.3d8p-33;
  float ub = r, lb = r - r*eps;
  if(__builtin_expect(ub != lb, 1)){
    if(__builtin_expect(ux<=0x79e7526eu, 0)){
      if(t.u == 0x3b429d37u) return 0x1.00870ap+0f - 0x1p-25f;
      if(t.u == 0xbcf3a937u) return 0x1.f58d62p-1f - 0x1p-26f;
      if(t.u == 0xb8d3d026u) return 0x1.fff6d2p-1f + 0x1p-26f;
    }
    static const double c[] =
      {0x1.62e42fefa39efp-1, 0x1.ebfbdff82c58fp-3, 0x1.c6b08d702e0edp-5, 0x1.3b2ab6fb92e5ep-7,
       0x1.5d886e6d54203p-10, 0x1.430976b8ce6efp-13};
    r = sv.f + (sv.f*h)*((c[0] + h*c[1]) + h2*((c[2] + h*c[3]) + h2*(c[4] + h*c[5])));
    ub = r;
  }
  return ub;
}

constexpr size_t kElems = 1ULL << 32;
constexpr size_t kSms = 132;

__global__ void CheckEx2(float* out) {
    for (
        size_t xx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        xx < kElems;
        xx += (size_t)blockDim.x * gridDim.x
    ) {
        unsigned xx_int = xx;
        float xx_float;
        memcpy(&xx_float, &xx_int, sizeof(float));

        float xx_out;
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(xx_out) : "f"(xx_float));
        out[xx] = xx_out;
    }
}

bool are_same(float x, float y) {
    uint32_t xb, yb;
    memcpy(&xb, &x, sizeof xb);
    memcpy(&yb, &y, sizeof y);
    return xb == yb;
}

int main() {
    float* out;
    auto status = cudaMalloc(&out, kElems * sizeof(float));
    if (status != cudaSuccess) {
        fprintf(stderr, "failed to malloc: %d\n", (int)status);
        exit(1);
    }

    CheckEx2<<<kSms, 1024>>>(out);

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "failed to launch: %d\n", (int)status);
        exit(1);
    }

    float* host_out = (float*)malloc(kElems * sizeof(float));
    status = cudaMemcpy(host_out, out, kElems * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "failed to memcpy: %d\n", (int)status);
        exit(1);
    }

    size_t diff_count = 0;
    for (size_t ii = 0; ii < kElems; ++ii) {
        if (ii % 1000000 == 0) {
            // fprintf(stderr, "%zu/%zu\n", ii, kElems);
        }

        float xx;
        memcpy(&xx, &ii, sizeof(float));
        // const auto ref_res = cr_exp2f(xx);
        const auto ref_res = ptxm_ex2_sm5x(xx);
        if (!are_same(ref_res, host_out[ii])) {
            printf("%g (%a): ref=%a hardware=%a\n", xx, xx, ref_res, host_out[ii]);
            ++diff_count;
        }
    }

    printf("DIFF COUNT: %zu\n", diff_count);
}
