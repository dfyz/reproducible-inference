#pragma once

#include <math.h>
#include <stdint.h>

constexpr size_t MAX_DIM = 64;

constexpr float INV_FREQS[MAX_DIM] = {
    0x1.000000p+00f,
    0x1.a11582p-01f,
    0x1.53c38cp-01f,
    0x1.14c708p-01f,
    0x1.c2ef78p-02f,
    0x1.6f56fap-02f,
    0x1.2b3dc4p-02f,
    0x1.e788c2p-03f,
    0x1.8d275cp-03f,
    0x1.438740p-03f,
    0x1.078d40p-03f,
    0x1.ad6338p-04f,
    0x1.5dc95ap-04f,
    0x1.1cf130p-04f,
    0x1.d03ccep-05f,
    0x1.7a2d08p-05f,
    0x1.341190p-05f,
    0x1.f5ea7ep-06f,
    0x1.98de92p-06f,
    0x1.4d1272p-06f,
    0x1.0f5386p-06f,
    0x1.ba0dd8p-07f,
    0x1.681adap-07f,
    0x1.2558fep-07f,
    0x1.ddee9ep-08f,
    0x1.8554ecp-08f,
    0x1.3d2804p-08f,
    0x1.025c6ap-08f,
    0x1.a4ee40p-09f,
    0x1.56e5b8p-09f,
    0x1.175482p-09f,
    0x1.c71820p-10f,
    0x1.72ba42p-10f,
    0x1.2e0046p-10f,
    0x1.ec07d4p-11f,
    0x1.90d10cp-11f,
    0x1.46831ap-11f,
    0x1.09fb7ep-11f,
    0x1.b15902p-12f,
    0x1.610332p-12f,
    0x1.1f91f0p-12f,
    0x1.d484dep-13f,
    0x1.7da9e8p-13f,
    0x1.36e8eap-13f,
    0x1.fa8b86p-14f,
    0x1.9ca3eap-14f,
    0x1.5024d6p-14f,
    0x1.11d41ep-14f,
    0x1.be2188p-15f,
    0x1.6b6d0ep-15f,
    0x1.280d96p-15f,
    0x1.e25700p-16f,
    0x1.88ec20p-16f,
    0x1.4014d2p-16f,
    0x1.04be68p-16f,
    0x1.a8d010p-17f,
    0x1.5a0f50p-17f,
    0x1.19e802p-17f,
    0x1.cb4a9ap-18f,
    0x1.76258ep-18f,
    0x1.30c94ep-18f,
    0x1.f09184p-19f,
    0x1.948360p-19f,
    0x1.4985fep-19f,
};

float sincos_approx(float x, bool is_cosine) {
    float k = rintf(x * 0x1.45f306p-1f);

    x = fmaf(k, -0x1.921fb4p+00f, x);
    x = fmaf(k, -0x1.4442d0p-24f, x);
    x = fmaf(k, -0x1.84698ap-48f, x);

    uint32_t k_int = (uint32_t)k + is_cosine;
    bool  even = k_int % 2 == 0;
    float z    = even ? x : 1.0f;
    float x_sq = x * x;

    float res = even
              ? -0x1.9a82a6p-13f
              : fmaf(0x1.9758p-16f, x_sq, -0x1.6c0fdap-10f);
    res = fmaf(res, x_sq, even ? 0x1.110bc8p-7f : 0x1.555576p-5f);
    res = fmaf(res, x_sq, even ? -0x1.55555p-3f : -0x1.fffffep-2f);
    res = fmaf(res, x_sq * z, z);

    if (k_int % 4 >= 2) {
        res = -res;
    }

    return res;
}

void rotate(float x, float y, float pos, size_t head_pos, float* out_x, float* out_y) {
    float angle  = pos * INV_FREQS[head_pos];
    float cosine = sincos_approx(angle, true);
    float   sine = sincos_approx(angle, false);

    *out_x = fmaf(x, cosine, -y*sine);
    *out_y = fmaf(y, cosine, +x*sine);
}
