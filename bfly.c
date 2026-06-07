#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define ROWS 8
#define COLS 16
float ORIG[ROWS][COLS];
float CUR[ROWS][COLS];
float TMP[COLS];

void shfl_bfly(float* orig_row, float* row, size_t to_xor, bool fma) {
    for (size_t cc = 0; cc < COLS; ++cc) {
        float old = row[cc];
        float new = row[cc ^ to_xor];

        if (fma) {
            TMP[cc] = fmaf(orig_row[cc], orig_row[cc], new);
        } else {
            TMP[cc] = old + new;
        }
    }

    memcpy(row, TMP, sizeof(TMP));
}


int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Provide an argument\n");
        return 1;
    }

    FILE* f = fopen(argv[1], "rb");
    if (f == NULL) {
        fprintf(stderr, "Failed to open\n");
        return 1;
    }

    if (fread(ORIG, sizeof(float), ROWS*COLS, f) != ROWS*COLS) {
        fprintf(stderr, "Failed to read\n");
        return 1;
    }
    fclose(f);

    memcpy(CUR, ORIG, sizeof(CUR));
    for (size_t rr = 0; rr < ROWS; ++rr) {
        float* orig_row = ORIG[rr];
        float* cur_row = CUR[rr];
        for (size_t cc = 0; cc < COLS; ++cc) {
            cur_row[cc] = cur_row[cc] * cur_row[cc];
        }

        shfl_bfly(orig_row, cur_row, 8, true);
        shfl_bfly(orig_row, cur_row, 4, false);
        shfl_bfly(orig_row, cur_row, 2, false);
        shfl_bfly(orig_row, cur_row, 1, false);

        printf("%1.8e ", cur_row[0]);
        for (size_t cc = 1; cc < COLS; ++cc) {
            if (cur_row[cc] != cur_row[0]) {
                printf("%1.8e ", cur_row[cc]);
                cur_row[0] = cur_row[cc];
            }
        }
        printf("\n");
    }
}
