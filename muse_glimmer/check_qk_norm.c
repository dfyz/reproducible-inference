#include "utils/bf16.h"
#include "utils/load_file.h"

#include "../ptx_math_rsqrt.h"

#include <math.h>
#include <stdio.h>

#include <err.h>

#ifndef N_ROWS
#error "Define N_ROWS"
#endif

constexpr size_t NUM_Q_HEADS      = 32;
constexpr size_t NUM_K_HEADS      = 2;
constexpr size_t HEAD_SIZE        = 128;
constexpr float  EPS              = 1e-5f;

constexpr size_t WARP_SIZE        = 32;
constexpr float  ELEMS_PER_THREAD = 4;

struct InOuts {
    bf16 q[N_ROWS][NUM_Q_HEADS][HEAD_SIZE];
    bf16 k[N_ROWS][NUM_K_HEADS][HEAD_SIZE];

    bf16 out_q[N_ROWS][NUM_Q_HEADS][HEAD_SIZE];
    bf16 out_k[N_ROWS][NUM_K_HEADS][HEAD_SIZE];
};

float sum_row(const bf16 head[HEAD_SIZE]) {
    float warp_vals[WARP_SIZE];
    for (size_t cc = 0, warp = 0; cc < HEAD_SIZE; cc += ELEMS_PER_THREAD, ++warp) {
        float acc = 0.0f;
        for (size_t off = 0; off < ELEMS_PER_THREAD; ++off) {
            float val = to_float(head[cc + off]);
            acc = fmaf(val, val, acc);
        }
        warp_vals[warp] = acc;
    }

    for (size_t bit = WARP_SIZE/2; bit > 0; bit >>= 1) {
        for (size_t cc = 0; cc < bit; ++cc) {
            warp_vals[cc] += warp_vals[cc ^ bit];
        }
    }

    return warp_vals[0];
}

void check_head(const char* label, size_t rr, size_t hh, bf16 head[HEAD_SIZE], const bf16 ref_out[HEAD_SIZE]) {
    float sum = sum_row(head);
    float rstd = ptxm_rsqrt_sm5x(fmaf(sum, 1.0f/HEAD_SIZE, EPS));
    for (size_t cc = 0; cc < HEAD_SIZE; ++cc) {
        float ref = to_float(ref_out[cc]);
        ref = to_float(to_bf16(ref));

        float our = to_float(head[cc]) * rstd;
        our = to_float(to_bf16(our));

        if (ref != our) {
            printf("%s (%zu, %zu): ref=%a, our=%a\n", label, rr, hh, ref, our);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        errx(42, "Provide a path to the file with inputs/outputs");
    }

    struct InOuts* io = load_file(argv[1], sizeof(struct InOuts));

    for (size_t rr = 0; rr < N_ROWS; ++rr) {
        for (size_t hh = 0; hh < NUM_Q_HEADS; ++hh) {
            if (hh < NUM_K_HEADS) {
                check_head("k", rr, hh, io->k[rr][hh], io->out_k[rr][hh]);
            }
            check_head("q", rr, hh, io->q[rr][hh], io->out_q[rr][hh]);
        }
    }
}