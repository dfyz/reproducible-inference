// gcc -std=c23 -O2 -march=native -DN_ROWS=8192 -lm -o check_fused_sigmoid_mul check_fused_sigmoid_mul.c

#include "utils/bf16.h"
#include "utils/ex2.h"
#include "utils/load_file.h"

#include "../ptx_math_recip.h"

#include <err.h>
#include <stdio.h>

#ifndef N_ROWS
#error "Define N_ROWS"
#endif

constexpr size_t DIM = 4096;
constexpr float  LOG2_E = 0x1.715476p+0f;

struct InOuts {
    bf16 attn[N_ROWS][DIM];
    bf16 out [N_ROWS][DIM];
    bf16 gate[N_ROWS][DIM];
};

float sigmoid(float x) {
    return ptxm_rcp_sm5x(1.0f + ex2(LOG2_E * -x));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        errx(42, "Provide a path to the file with inputs/outputs");
    }

    struct InOuts* io = load_file(argv[1], sizeof(struct InOuts));

    for (size_t rr = 0; rr < N_ROWS; ++rr) {
        for (size_t cc = 0; cc < DIM; ++cc) {
            float ref = to_float(io->out[rr][cc]);

            float attn = to_float(io->attn[rr][cc]);
            float gate = to_float(io->gate[rr][cc]);
            float our = attn * sigmoid(gate);
            our = to_float(to_bf16(our));

            if (ref != our) {
                printf("%zu, %zu: %a vs. %a\n", rr, cc, ref, our);
            }
        }
    }
}
