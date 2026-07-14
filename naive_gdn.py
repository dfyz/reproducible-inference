import math
import torch
import sys

from tqdm import trange


def do_recurrent(ins, seq_len, n_v_heads, qk_ratio, head_dim, scale):
    full_state = ins['initial_state'].clone()
    out = torch.empty((seq_len, n_v_heads, head_dim), dtype=torch.float32)

    for hh in trange(n_v_heads):
        qk_hh = hh // qk_ratio
        head_state = full_state[0, hh]
        for tt in range(seq_len):
            q, k, v = ins['q'][tt, qk_hh].float(), ins['k'][tt, qk_hh].float(), ins['v'][tt, hh].float()
            a, b = ins['g'][tt, hh], ins['beta'][tt, hh]

            head_state *= a
            head_state += b * (v - head_state @ k).unsqueeze(1) @ k.unsqueeze(0)
            out[tt, hh] = scale * (head_state @ q)

    return out.to(torch.bfloat16), full_state


CHUNK_SIZE = 64


def do_chunked(ins, seq_len, n_v_heads, qk_ratio, head_dim, scale):
    full_state = ins['initial_state'].clone()
    out = torch.empty((seq_len, n_v_heads, head_dim), dtype=torch.float32)

    full_q = ins['q'].repeat_interleave(qk_ratio, dim=1)
    full_k = ins['k'].repeat_interleave(qk_ratio, dim=1)
    causal_mask = torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE))

    for hh in trange(n_v_heads):
        cur_state = full_state[0, hh]

        for tt in range(0, seq_len, CHUNK_SIZE):
            chunk_size = min(CHUNK_SIZE, seq_len - tt)
            chunk = slice(tt, tt + chunk_size)
            q, k, v = full_q[chunk, hh].float(), full_k[chunk, hh].float(), ins['v'][chunk, hh].float()
            a, b = ins['g'][chunk, hh], ins['beta'][chunk, hh]

            left_cumprod = torch.cumprod(a, dtype=torch.double, dim=0)
            full_decay = left_cumprod[-1]
            right_cumprod = full_decay / left_cumprod

            diag_b = torch.diag(b)
            decay_left = torch.diag(left_cumprod).float()
            decay_right = torch.diag(right_cumprod).float()

            kk = k @ k.T
            gamma = torch.tril(left_cumprod.unsqueeze(1) / left_cumprod.unsqueeze(0)).float()

            def tril_invert_scale_b(mat):
                return torch.linalg.inv(
                    torch.eye(chunk_size) + torch.tril(diag_b @ mat, -1)
                ) @ diag_b

            u = tril_invert_scale_b(gamma * kk) @ v
            w = decay_left @ tril_invert_scale_b(kk) @ k
            new_v = u - w @ cur_state.T

            out[chunk, hh] = scale * (decay_left @ q @ cur_state.T + ((q @ k.T) * gamma) @ new_v)
            full_state[0, hh] = full_decay * cur_state + new_v.T @ decay_right @ k

    return out.to(torch.bfloat16), full_state


if __name__ == '__main__':
    ins, (ref_out, ref_state) = torch.load(sys.argv[1], map_location='cpu')
    seq_len, n_v_heads, head_dim = ref_out.shape
    qk_ratio = n_v_heads // ins['q'].shape[1]
    scale = 1 / math.sqrt(head_dim)

    chunk_out, chunk_state = do_chunked(ins, seq_len, n_v_heads, qk_ratio, head_dim, scale)
    print('Max absolute chunked error', abs(chunk_out - ref_out).max(), abs(chunk_state - ref_state).max())

    rec_out, rec_state = do_recurrent(ins, seq_len, n_v_heads, qk_ratio, head_dim, scale)
    print('Max absolute recurrent error', abs(rec_out - ref_out).max(), abs(rec_state - ref_state).max())
