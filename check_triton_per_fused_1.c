#include <err.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROWS 32
#define COLS 128
// 1e-6
#define EPS 0x1.0c6f7ap-20f

typedef uint16_t bf16;

struct InOuts {
    bf16 norm_input[ROWS][COLS];
    bf16 silu_input[ROWS][COLS];
    bf16 norm_weight[COLS];
    float ref_out[ROWS][COLS];
};

union flint {
    float f;
    uint32_t i;
};

float to_float(bf16 x) {
    union flint res = {.i = ((uint32_t)x) << 16};
    return res.f;
}

struct InOuts* load(const char* file_name) {
    // We rely on the OS unmapping the file and closing the FD.
    int fd = open(file_name, O_RDONLY);
    if (fd < 0) {
        err(EXIT_FAILURE, "Failed to open %s", file_name);
    }

    void* res = mmap(NULL, sizeof(struct InOuts), PROT_READ, MAP_SHARED, fd, 0);
    if (res == MAP_FAILED) {
        err(EXIT_FAILURE, "Failed to mmap");
    }

    return (struct InOuts*)res;
}

float silu(float x) {
    return x / (1.0f + expf(-x));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        errx(EXIT_FAILURE, "Provide a path to the file with inputs/outputs");
    }

    struct InOuts* in_outs = load(argv[1]);

    for (size_t rr = 0; rr < ROWS; ++rr) {
        float sq_sum = 0.0f;
        for (size_t cc = 0; cc < COLS; ++cc) {
            float val = to_float(in_outs->norm_input[rr][cc]);
            sq_sum += val * val;
        }
        float rms = sqrtf(sq_sum / COLS + EPS);

        for (size_t cc = 0; cc < COLS; ++cc) {
            float val = to_float(in_outs->norm_input[rr][cc]);
            float weight = to_float(in_outs->norm_weight[cc]);
            float silu_res = silu(to_float(in_outs->silu_input[rr][cc]));
            float ours = val / rms * weight * silu_res;
            float ref = in_outs->ref_out[rr][cc];

            if (ours != ref) {
                printf("(%zu, %zu): %1.8e (%a) vs. %1.8e (%a)\n", rr, cc, ours, ours, ref, ref);
            }
        }
    }
}