#include <err.h>
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    FILE* f = fopen("inv_freq.bin", "rb");
    if (f == NULL) errx(1, "fopen");

    const size_t NUM_F = 32;
    float ref[NUM_F];
    if (fread(ref, sizeof(float), NUM_F, f) != NUM_F) {
        errx(1, "fread");
    }
    for (size_t ii = 0; ii < NUM_F; ++ii) {
        float ours = 1.0f / powf(1e7f, (float)(2 * ii) / (float)(2 * NUM_F));
        printf("%zu\t%e (%a) vs. %e (%a)\n", ii, ref[ii], ref[ii], ours, ours);
        if (ref[ii] != ours) {
            printf("MISMATCH\n");
        }
    }
}
