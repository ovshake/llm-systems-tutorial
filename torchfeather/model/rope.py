import math 

import torch 

from torchfeather.model.model_args import DeepSeekV3ModelArgs 



def precompute_freq_cis(args: DeepSeekV3ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    base = args.rope_theta
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow 
    factor = args.rope_factor 


    def find_correction_dim(
        num_rotations: float, total_dim: int, base: float, max_seq_len: int
    ) -> float:
        return (
            total_dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
        )

    
    def find_correction_range(
        low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
    ) -> tuple[int, int]:
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    if seqlen > args.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2) 
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)


    freqs = torch.outer(t, freqs) 

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 

    return freqs_cis 

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # X: [B, S, H, D] -> [B, S, H, D // 2, 2] 
    # freqs_cis: [S, D/2]
    dtype = x.dtype 
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2)) # [B, S, H, D] -> [B, S, H, D // 2, 2] -> [B, S, H, D // 2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # [S, D/2] -> [1, S, 1, D/2]

    y = torch.view_as_real(x * freqs_cis).flatten(3) 

    return y.to(dtype)
