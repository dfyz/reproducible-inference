#include "utils/bf16.h"
#include "utils/ex.h"
#include "utils/load_file.h"
#include "utils/sigmoid.h"

#include "../ptx_math_recip.h"

#include <err.h>
#include <stdio.h>

#ifndef N_ROWS
#error "Define N_ROWS"
#endif

constexpr size_t DIM = 19968;

struct InOuts {
    bf16 inp[N_ROWS][2*DIM];
    bf16  out[N_ROWS][DIM];
};

int main(int argc, char** argv) {
    if (argc != 2) {
        errx(42, "Provide a path to the file with inputs/outputs");
    }

    struct InOuts* io = load_file(argv[1], sizeof(struct InOuts));

    for (size_t rr = 0; rr < N_ROWS; ++rr) {
        for (size_t cc = 0; cc < DIM; ++cc) {
            float ref = to_float(io->out[rr][cc]);

            float to_act  = to_float(io->inp[rr][cc]);
            float to_gate = to_float(io->inp[rr][cc + DIM]);

            float our = to_act * sigmoid_fast(to_act) * to_gate;
            our = to_float(to_bf16(our));

            if (ref != our) {
                printf("%zu, %zu: %a vs. %a\n", rr, cc, ref, our);
            }
        }
    }
}
