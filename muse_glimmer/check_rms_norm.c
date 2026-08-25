// gcc -std=c23 -O2 -g -march=native -lm -DN_ROWS=8192 -DEPS=1e-05f -o check_rms_norm check_rms_norm.c

#include "utils/bf16.h"
#include "utils/load_file.h"

// TODO: rewrite MUFU.RSQRT properly.
#include "../ptx_math_rsqrt.h"

#include <stdint.h>
#include <stdio.h>

#include <err.h>

#if !defined(N_ROWS) || !defined(EPS)
#error "Define N_ROWS and EPS"
#endif

constexpr size_t DIM                     = 6656;
constexpr size_t WARP_SIZE               = 32;
constexpr size_t N_WARPS                 = 4;
constexpr size_t CONTIG_ELEMS_PER_THREAD = 8;

struct InOuts {
    bf16 inp[N_ROWS][DIM];
    bf16 out[N_ROWS][DIM];
    bf16 w  [DIM];
};

typeof((struct InOuts){0}.out) OUR_OUT;

float warp_reduce(float vals[WARP_SIZE]) {
    // Yeah, flashinfer really iterates over offsets in increasing order.
    for (size_t bit = 1; bit < WARP_SIZE; bit <<= 1) {
        for (size_t ii = 0; ii < WARP_SIZE; ++ii) {
            size_t adj = ii ^ bit;
            if (ii < adj) {
                float res = vals[ii] + vals[adj];
                vals[ii] = vals[adj] = res;
            }
        }
    }
    return vals[0];
}

float sum_row(const bf16 row[DIM]) {
    float warp_vals[N_WARPS];
    float lane_vals[WARP_SIZE] = {};
    for (size_t warp = 0; warp < N_WARPS; ++warp) {
        for (size_t lane = 0; lane < WARP_SIZE; ++lane) {
            float lane_acc = 0.0f;
            for (
                size_t start = (warp*WARP_SIZE + lane)*CONTIG_ELEMS_PER_THREAD;
                start        <  DIM;
                start        += N_WARPS*WARP_SIZE*CONTIG_ELEMS_PER_THREAD
            ) {
                for (size_t elem = 0; elem < CONTIG_ELEMS_PER_THREAD; ++elem) {
                    float val = to_float(row[start + elem]);
                    lane_acc = fmaf(val, val, lane_acc);
                }
            }
            lane_vals[lane] = lane_acc;
        }
        warp_vals[warp] = warp_reduce(lane_vals);
    }

    return (warp_vals[0] + warp_vals[1]) + (warp_vals[2] + warp_vals[3]);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        errx(42, "Provide a path to the file with inputs/outputs");
    }

    struct InOuts* io = load_file(argv[1], sizeof(struct InOuts));

    for (size_t rr = 0; rr < N_ROWS; ++rr) {
        auto inp_row = io->inp[rr];
        auto ref_out = io->out[rr];

        float sum = sum_row(inp_row);
        float rstd = ptxm_rsqrt_sm5x(sum / DIM + EPS);

        for (size_t cc = 0; cc < DIM; ++cc) {
            float ref = to_float(ref_out[cc]);
            ref = to_float(to_bf16(ref));

            float our = to_float(inp_row[cc]) * rstd * to_float(io->w[cc]);
            our = to_float(to_bf16(our));

            if (ref != our) {
                printf("(%zu, %zu): ref=%a, out=%a\n", rr, cc, ref, our);
            }
        }
    }
}
