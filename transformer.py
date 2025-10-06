import torch
from torch import nn

# Build Map:

# Transformer = Tokenizer -> Embedder -> Attention Block (Nx) -> Norm -> Linear -> Softmax
# Tokenizer = BPE / import
# Embedder = random weights
# Attn Block = RMSNorm + MHA + (add residual x) + RMSNorm + FFN (MLP) + (add residual x)
# RMSNorm = x / (RMS(x) + eps)
# MHA = project x to W_i, Q_i, K_i and ship to Attn Head i -> concat results *W_o -> out
# Attn Head i = [Q, K = RoPE(Q, K)] -> softmax((QK^T) / sqrt(d/H) + M) V -> out
# MLP = (linear -> activation) (Nx) -> out

# TODO:
#   KV Cache!
#   fuse QKV
#   gradient clipping
#   Configs
#   hooks?


# CONFIGS:
# d = dim(embedding), T = #toks, H = #heads, B = batch size
# d = 1024
# T = 200
# H = 8
# B = 1

# d_ff = 2048 # d_ff = dim(FFN hidden layers)
# MAXT = 4098 # max tokens
# V = 4098 # vocab size
# RoPE_base = 10000 # RoPE base (just need sufficiently large?)

# f32 = torch.float32

class FeedForwardNetwork(nn.Module):
    # TODO:
    #   SwiGLU X
    #   support n layers? X (no need)

    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d, d_ff)
        self.silu = nn.SiLU()
        self.up = nn.Linear(d, d_ff)
        self.down = nn.Linear(d_ff, d)
        
    def forward(self, x):
        out = self.silu(self.gate(x)) * self.up(x)     # SwiGLU : d_ff ~ 2.7d
        out = self.down(out)
        return out
    
class MultiHeadedAttention(nn.Module):
    # TODO:
    #   RoPE X
    #   KV cache
    #   update mask? X
    #   matmuls? X
    #   require_grad?
    #   fuse QKV
    
    def __init__(self, d: int, H: int, MAXT: int, p_drop: float) -> None:
        super().__init__()
        # assuming q = v = d/H
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(p_drop)

        self.d = d
        self.H = H
        if d % H: raise ValueError('#heads must divide dim(embedding)')
        self.d_h = d // H

        # causal mask
        self.register_buffer('mask', torch.triu(torch.ones((1, 1, MAXT, MAXT), dtype=torch.bool), 1))

        # angles : (1 x 1 x MAXT x d_h/2)
        # thetas : (d_h/2)
        if self.d_h & 1: raise ValueError('dim(head) must be even')
        RoPE_base = 10000   # just need sufficiently large?
        self.register_buffer('theta', RoPE_base ** (-2 * torch.arange(self.d_h // 2, dtype=torch.float32) / self.d_h))

    def forward(self, X):
        # X : (B x T x d)
        B, T, d = X.shape
        if d != self.d: raise ValueError('dim(input) does not match dim(model)')

        # Q, K, V : (B x T x d)
        Q, K, V = self.W_q(X), self.W_k(X), self.W_v(X)

        # reshape into heads [row: [tok: [dh: [], dh: [], ... ], tok: []] row: ]
        # [row: [dh: [tok: [], tok: []], dh: [tok: [], tok: []] row: ]
        # Q_h, K_h, V_h : (B x H x T x d_h)
        Q_h = torch.reshape(Q, (B, T, self.H, self.d_h)).transpose(1, 2)
        K_h = torch.reshape(K, (B, T, self.H, self.d_h)).transpose(1, 2)
        V_h = torch.reshape(V, (B, T, self.H, self.d_h)).transpose(1, 2)
        if not (Q_h.dtype == K_h.dtype == V_h.dtype): raise TypeError('QKV dtypes are mismatched')

        # RoPE:
        #   x_(i,2j) = cos(angles[i]) * x_(i,2j) - sin(angles[i]) * x_(i,2j+1)
        #   x_(i,2j+1) = sin(angles[i]) * x_(i,2j) + cos(angles[i]) * x_(i,2j+1)
        pos = torch.arange(T, dtype=torch.float32, device=X.device)     # because built at runtime
        angles = torch.outer(pos, self.theta)[None,None,:,:]
        cos = torch.cos(angles).to(Q_h.dtype)   # Q_h/K_h.dtype
        sin = torch.sin(angles).to(K_h.dtype)

        Q_00, Q_10 = Q_h[:,:,:,0::2], Q_h[:,:,:,1::2]
        K_00, K_10 = K_h[:,:,:,0::2], K_h[:,:,:,1::2]
        Q_01 = cos * Q_00 - sin * Q_10
        Q_11 = sin * Q_00 + cos * Q_10
        K_01 = cos * K_00 - sin * K_10
        K_11 = sin * K_00 + cos * K_10

        Q_r = torch.reshape(torch.stack((Q_01, Q_11), dim=-1).contiguous(), (B, self.H, T, self.d_h))
        K_r = torch.reshape(torch.stack((K_01, K_11), dim=-1).contiguous(), (B, self.H, T, self.d_h))


        # pattern : (B x H x T x T)
        # heads : (B x T x H x d_h)
        # out : (B x T x d)
        scores = Q_r @ K_r.transpose(-2, -1) * self.d_h ** -0.5     # ** -0.5 vs sqrt
        scores = scores.masked_fill(self.mask[:,:,:T,:T], torch.finfo(scores.dtype).min)
        pattern = self.dropout(torch.softmax(scores, -1))    # dropout

        heads = (pattern @ V_h).transpose(1, 2).contiguous()    # contiguous mem
        out = torch.reshape(heads, (B, T, d))
        return self.W_o(out)
    
class AttentionBlock(nn.Module):
    def __init__(self, d: int, H: int, MAXT: int, d_ff: int, p_drop: float) -> None:
        super().__init__()
        self.MHA = MultiHeadedAttention(d, H, MAXT, p_drop)
        self.FFN = FeedForwardNetwork(d, d_ff)
        self.RMS1 = nn.RMSNorm(d)
        self.RMS2 = nn.RMSNorm(d)
        self.dropout = nn.Dropout(p_drop)
    
    def forward(self, X):
        # ... : (B x T x d)
        out = X + self.dropout(self.MHA(self.RMS1(X)))
        out = out + self.dropout(self.FFN(self.RMS2(out)))
        return out
    
class Transformer(nn.Module):
    # TODO:
    #   dropout X
    #   input params X
    #   LM head? (bias optional) X
    
    def __init__(self, dim_model: int, num_layers: int, num_heads: int, max_tokens: int, vocab_size: int, dim_ff: int, p_drop: float) -> None:
        super().__init__()
        self.d = dim_model
        self.L = num_layers
        self.H = num_heads
        self.MAXT = max_tokens
        self.V = vocab_size
        self.d_ff = dim_ff
        self.p_drop = p_drop

        self.embedding = nn.Embedding(self.V, self.d)
        self.blocks = nn.ModuleList([AttentionBlock(self.d, self.H, self.MAXT, self.d_ff, self.p_drop) for _ in range(self.L)])
        self.RMS = nn.RMSNorm(self.d)

    def forward(self, X):
        B, T = X.shape
        if T > self.MAXT: raise ValueError(f'too many tokens; max {self.MAXT} tokens')
        if X.dim() != 2: raise ValueError(f'wrong number of dimensions: requires 2, received {X.dim()}')
        if X.dtype != torch.long: raise TypeError(f'input is wrong type: requires torch.long, received {X.dtype}')
        
        out = self.embedding(X)
        for block in self.blocks:
            out = block(out)
        out = self.RMS(out)
        out = out @ self.embedding.weight.T
        return out