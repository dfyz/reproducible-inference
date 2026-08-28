// gcc -std=c23 -march=native -lm -O2 -o muse_glimmer muse_glimmer.c -pedantic

#include "utils/bf16.h"
#include "utils/rope.h"
#include "utils/tc.h"

#include "../ptx_math_rsqrt.h"

#include <err.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

constexpr size_t DIM              = 6656;
constexpr size_t INTERMEDIATE_DIM = DIM * 3;
constexpr size_t HEAD_DIM         = 128;

constexpr size_t N_Q_HEADS        = 32;
constexpr size_t N_KV_HEADS       = 2;

constexpr size_t VOCAB_SIZE       = 202048;

// MODEL LOADING
struct layer {
    bf16 input_ln    [DIM];
    bf16 mlp_down    [DIM]             [INTERMEDIATE_DIM];
    bf16 mlp_gate    [INTERMEDIATE_DIM][DIM];
    bf16 mlp_up      [INTERMEDIATE_DIM][DIM];
    bf16 post_attn_ln[DIM];
    bf16 post_ff_ln  [DIM];
    bf16 pre_ff_ln   [DIM];
    bf16 attn_gate   [N_Q_HEADS] [HEAD_DIM] [DIM];
    bf16 attn_k      [N_KV_HEADS][HEAD_DIM] [DIM];
    bf16 attn_o      [DIM]       [N_Q_HEADS][HEAD_DIM];
    bf16 attn_q      [N_Q_HEADS] [HEAD_DIM] [DIM];
    bf16 attn_v      [N_KV_HEADS][HEAD_DIM] [DIM];
};

// layers 0 to 44 (inclusive)
constexpr size_t N_LAYERS_PART_1 = 45;
struct layers_part_1 {
    bf16 embeds[VOCAB_SIZE][DIM];
    struct layer layers[N_LAYERS_PART_1];
};

constexpr size_t N_LAYERS_PART_2 = 7;
struct layers_part_2 {
    bf16 lm_head[VOCAB_SIZE][DIM];
    // layers 45 to 51 (inclusive)
    struct layer layers[N_LAYERS_PART_2];
};

constexpr size_t N_LAYERS = N_LAYERS_PART_1 + N_LAYERS_PART_2;

struct model {
    struct layers_part_1* p1;
    struct layers_part_2* p2;
};

void* load_st(const char* file_name, void* addr, size_t n_bytes) {
    // We rely on the OS unmapping the file and closing the FD.
    int fd = open(file_name, O_RDONLY);
    if (fd < 0) {
        err(1, "failed to open %s", file_name);
    }

    uint64_t h_len;
    if (read(fd, &h_len, sizeof(h_len)) != sizeof(h_len)) {
        err(2, "failed to read the header length from %s", file_name);
    }

    char* res = mmap(addr, n_bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (res == MAP_FAILED) {
        err(3, "failed to mmap %s", file_name);
    }
    return res + sizeof(h_len) + h_len;
}

struct model load_model() {
    return (struct model) {
        .p1 = load_st("model-00001-of-00002.safetensors", nullptr, sizeof(struct layers_part_1)),
        .p2 = load_st("model-00002-of-00002.safetensors", nullptr, sizeof(struct layers_part_2)),
    };
}

// COMMON CONSTANTS AND GLOBAL DATA
constexpr size_t MAX_SEQ_LEN = 16384;
bf16 K_CACHE[N_LAYERS][MAX_SEQ_LEN][N_Q_HEADS] [HEAD_DIM];
bf16 V_CACHE[N_LAYERS][MAX_SEQ_LEN][N_KV_HEADS][HEAD_DIM];

constexpr float PRE_NORM_EPS  = 1e-05f;
constexpr float POST_NORM_EPS = 1e-08f;

typedef bf16 vec[DIM];

constexpr size_t WARP_SIZE = 32;

// RMSNorm
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

float sum_row(const vec x) {
    constexpr size_t N_WARPS                 = 4;
    constexpr size_t CONTIG_ELEMS_PER_THREAD = 8;

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
                    float val = to_float(x[start + elem]);
                    lane_acc = fmaf(val, val, lane_acc);
                }
            }
            lane_vals[lane] = lane_acc;
        }
        warp_vals[warp] = warp_reduce(lane_vals);
    }

    return (warp_vals[0] + warp_vals[1]) + (warp_vals[2] + warp_vals[3]);
}

void rms_norm(vec x, float eps, const bf16 w[DIM]) {
    float sum = sum_row(x);
    float rstd = ptxm_rsqrt_sm5x(sum / DIM + eps);

    for (size_t ii = 0; ii < DIM; ++ii) {
        float normed = to_float(x[ii]) * rstd * to_float(w[ii]);
        x[ii] = to_bf16(normed);
    }
}

// QK Norm
float sum_row_qk(const bf16 head[HEAD_DIM]) {
    constexpr size_t ELEMS_PER_THREAD = 4;

    float warp_vals[WARP_SIZE];
    for (size_t ii = 0, warp = 0; ii < HEAD_DIM; ii += ELEMS_PER_THREAD, ++warp) {
        float acc = 0.0f;
        for (size_t off = 0; off < ELEMS_PER_THREAD; ++off) {
            float val = to_float(head[ii + off]);
            acc = fmaf(val, val, acc);
        }
        warp_vals[warp] = acc;
    }

    for (size_t bit = WARP_SIZE/2; bit > 0; bit >>= 1) {
        for (size_t ii = 0; ii < bit; ++ii) {
            warp_vals[ii] += warp_vals[ii ^ bit];
        }
    }

    return warp_vals[0];
}

void qk_norm(bf16 head[HEAD_DIM]) {
    float sum = sum_row_qk(head);
    float rstd = ptxm_rsqrt_sm5x(fmaf(sum, 1.0f/HEAD_DIM, PRE_NORM_EPS));
    for (size_t ii = 0; ii < HEAD_DIM; ++ii) {
        head[ii] = to_bf16(to_float(head[ii]) * rstd);
    }
}

// FORWARD FOR ONE LAYER
void add(vec acc, const vec x) {
    for (size_t ii = 0; ii < DIM; ++ii) {
        acc[ii] = to_bf16(to_float(acc[ii]) + to_float(x[ii]));
    }
}

void rotate_head(size_t pos, bf16 head[HEAD_DIM]) {
    for (size_t ii = 0; ii < HEAD_DIM/2; ++ii) {
        float x = to_float(head[ii]);
        float y = to_float(head[ii + HEAD_DIM/2]);
        rotate(&x, &y, pos, ii);
        head[ii]              = to_bf16(x);
        head[ii + HEAD_DIM/2] = to_bf16(y);
    }
}

void attn(vec x, size_t pos, size_t layer_idx, const struct layer* l) {
    bf16 q[N_Q_HEADS][HEAD_DIM];
    auto k = K_CACHE[layer_idx][pos];
    auto v = V_CACHE[layer_idx][pos];

    for (size_t hh = 0; hh < N_Q_HEADS; ++hh) {
        for (size_t ii = 0; ii < HEAD_DIM; ++ii) {
            q[hh][ii] = to_bf16(tc_bf16_fp32(0.0f, x, l->attn_q[hh][ii], DIM));
        }
        qk_norm(q[hh]);
        rotate_head(pos, q[hh]);
    }

    for (size_t hh = 0; hh < N_KV_HEADS; ++hh) {
        for (size_t ii = 0; ii < HEAD_DIM; ++ii) {
            k[hh][ii] = to_bf16(tc_bf16_fp32(0.0f, x, l->attn_k[hh][ii], DIM));
            v[hh][ii] = to_bf16(tc_bf16_fp32(0.0f, x, l->attn_v[hh][ii], DIM));
        }
        qk_norm(k[hh]);
        rotate_head(pos, k[hh]);
    }

    // TODO: attention+gating+out projection
}

void mlp(vec x, const struct layer* l) {
    // TODO: mlp
}

void run_layer(vec x, size_t pos, size_t layer_idx, const struct layer* l) {
    vec residual;
    memcpy(residual, x, sizeof(residual));

    rms_norm(x, PRE_NORM_EPS, l->input_ln);
    attn    (x, pos, layer_idx, l);
    rms_norm(x, POST_NORM_EPS, l->post_attn_ln);
    add     (residual, x);

    rms_norm(x, PRE_NORM_EPS, l->pre_ff_ln);
    mlp     (x, l);
    rms_norm(x, POST_NORM_EPS, l->post_ff_ln);
    add     (x, residual);
}

int main() {
    auto model = load_model();

    // TODO: input actual tokens
    printf("%a\n", to_float(model.p2->layers[6].attn_gate[11][111][1111]));
}
