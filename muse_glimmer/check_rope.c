// gcc -std=c23 -O2 -g -march=native -lm -DN_ROWS=9654 -DSTART_POS=1 -o check_rope check_rope.c

#include "utils/bf16.h"
#include "utils/load_file.h"
#include "utils/rope.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <err.h>

#if !defined(N_ROWS) || !defined(START_POS)
#error "Define N_ROWS and START_POS"
#endif

constexpr size_t NUM_Q_HEADS      = 32;
constexpr size_t NUM_K_HEADS      = 2;
constexpr size_t HEAD_SIZE        = 128;

struct InOuts {
    bf16 q[N_ROWS][NUM_Q_HEADS][HEAD_SIZE];
    bf16 k[N_ROWS][NUM_K_HEADS][HEAD_SIZE];

    bf16 out_q[N_ROWS][NUM_Q_HEADS][HEAD_SIZE];
    bf16 out_k[N_ROWS][NUM_K_HEADS][HEAD_SIZE];
};

void rotate_head(float pos, const bf16 head[HEAD_SIZE], const bf16 ref_out[HEAD_SIZE]) {
    for (size_t head_pos = 0; head_pos < HEAD_SIZE/2; ++head_pos) {
        float x = to_float(head[head_pos]);
        float y = to_float(head[head_pos + HEAD_SIZE/2]);
        rotate(&x, &y, pos, head_pos);

        float our_x = to_float(to_bf16(x));
        float our_y = to_float(to_bf16(y));

        float ref_x = to_float(ref_out[head_pos]);
        float ref_y = to_float(ref_out[head_pos + HEAD_SIZE/2]);

        if (our_x != ref_x) {
            printf("x at %g, %zu: ref=%a, our=%a\n", pos, head_pos, ref_x, our_x);
        }

        if (our_y != ref_y) {
            printf("y at %g, %zu: ref=%a, our=%a\n", pos, head_pos, ref_y, our_y);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        errx(42, "Provide a path to the file with inputs/outputs");
    }

    struct InOuts* io = load_file(argv[1], sizeof(struct InOuts));

    for (size_t rr = 0; rr < N_ROWS; ++rr) {
        float pos = rr + START_POS;
        for (size_t hh = 0; hh < NUM_Q_HEADS; ++hh) {
            if (hh < NUM_K_HEADS) {
                rotate_head(pos, io->k[rr][hh], io->out_k[rr][hh]);
            }
            rotate_head(pos, io->q[rr][hh], io->out_q[rr][hh]);
        }
    }
}
