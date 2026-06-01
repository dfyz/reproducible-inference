#include "ptx_math_exp2f.h"

#include <fenv.h>
#include <math.h>
#include <string.h>

float ptx_expf(float x) {
    const float log2e = 0x1.715476p+0f; // 1.442695e+00
    float y = fmaf(x, 0x1.77313ap-8 /*5.724980e-03*/, 0.5f);
    y = fminf(1.0f, fmaxf(0.0f, y));
    fesetround(FE_DOWNWARD);
    y = fmaf(y, 252.0f, 12582913.0f);
    fesetround(FE_TONEAREST);

    float z = y - 12583039.0f;
    z = fmaf(x, log2e, -z);
    z = fmaf(x, 0x1.4ae0cp-26 /*1.925963e-08*/, z);

    unsigned y_int;
    memcpy(&y_int, &y, sizeof(unsigned));
    y_int <<= 23;
    memcpy(&y, &y_int, sizeof(float));

    return y * ptxm_ex2_sm5x(z);
}
