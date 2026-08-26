// gcc -std=c23 -O2 -march=native -o check_cos_sin_cache check_cos_sin_cache.c -lm

#include "utils/load_file.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <err.h>

constexpr size_t MAX_LEN = 1ull << 17;
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

struct InOuts {
    float cosines[MAX_LEN][MAX_DIM];
    float   sines[MAX_LEN][MAX_DIM];
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

int main(int argc, char** argv) {
    if (argc != 2) {
        errx(42, "Provide a path to the file with inputs/outputs");
    }

    struct InOuts* io = load_file(argv[1], sizeof(struct InOuts));

    for (size_t pos = 0.0f; pos < MAX_LEN; ++pos) {
        for (size_t dd = 0; dd < MAX_DIM; ++dd) {
            float angle = pos * INV_FREQS[dd];

            float ref_cos = io->cosines[pos][dd];
            float our_cos = sincos_approx(angle, true);
            if (ref_cos != our_cos) {
                printf("cosine at pos=%zu, dim=%zu: %a -> %a vs. %a\n", pos, dd, angle, ref_cos, our_cos);
            }

            float ref_sin = io->sines[pos][dd];
            float our_sin = sincos_approx(angle, false);
            if (ref_sin != our_sin) {
                printf("sine at pos=%zu, dim=%zu: %a -> %a vs. %a\n", pos, dd, angle, ref_sin, our_sin);
            }
        }
    }
}
