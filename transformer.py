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
#   DTYPE configs?
#   general dtype enforcement
#   remove global params


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
    
    def __init__(self, d: int, H: int, maxT: int) -> None:
        super().__init__()
        # assuming q = v = d/H
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)

        self.d = d
        self.H = H
        if d % H:
            raise Exception('#heads must divide dim(embedding)')
        self.d_h = d // H

        # causal mask
        self.register_buffer('mask', torch.triu(torch.ones((1, 1, maxT, maxT), dtype=bool), 1))

        # angles : (1 x 1 x MAXT x d_h/2)
        # thetas : (d_h/2)
        if self.d_h & 1:
            raise Exception('dim(head) must be even')
        RoPE_base = 10000   # just need sufficiently large?
        self.register_buffer('theta', RoPE_base ** (-2 * torch.arange(self.d_h // 2, dtype=torch.float32) / self.d_h))

    def forward(self, X):
        # X : (B x T x d)
        B, T, d = X.shape
        if d != self.d:
            raise Exception('dim(input) does not match dim(model)')

        # Q, K, V : (B x T x d)
        Q, K, V = self.W_q(X), self.W_k(X), self.W_v(X)

        # reshape into heads [row: [tok: [dh: [], dh: [], ... ], tok: []] row: ]
        # [row: [dh: [tok: [], tok: []], dh: [tok: [], tok: []] row: ]
        # Q_h, K_h, V_h : (B x H x T x d_h)
        Q_h = torch.reshape(Q, (B, T, self.H, self.d_h)).transpose(1, 2)
        K_h = torch.reshape(K, (B, T, self.H, self.d_h)).transpose(1, 2)
        V_h = torch.reshape(V, (B, T, self.H, self.d_h)).transpose(1, 2)


        # RoPE:
        #   x_(i,2j) = cos(angles[i]) * x_(i,2j) - sin(angles[i]) * x_(i,2j+1)
        #   x_(i,2j+1) = sin(angles[i]) * x_(i,2j) + cos(angles[i]) * x_(i,2j+1)
        angles = torch.outer(torch.arange(T, dtype=torch.float32), self.theta)[None,None,:,:]
        cos = torch.cos(angles).to(Q_h.dtype)   # Q_h/K_h.dtype
        sin = torch.sin(angles).to(K_h.dtype)

        Q_00, Q_10 = Q_h[:,:,:,0::2], Q_h[:,:,:,1::2]
        K_00, K_10 = K_h[:,:,:,0::2], K_h[:,:,:,1::2]
        Q_01 = cos * Q_00 - sin * Q_10
        Q_11 = sin * Q_00 + cos * Q_10
        K_01 = cos * K_00 - sin * K_10
        K_11 = sin * K_00 + cos * K_10

        Q_r = torch.reshape(torch.stack((Q_01, Q_11), dim=-1), (B, self.H, T, self.d_h))
        K_r = torch.reshape(torch.stack((K_01, K_11), dim=-1), (B, self.H, T, self.d_h))


        # pattern : (B x H x T x T)
        # heads : (B x T x H x d_h)
        # out : (B x T x d)
        scores = Q_r @ K_r.transpose(-2, -1) * self.d_h ** -0.5     # ** -0.5 vs sqrt
        scores = scores.masked_fill(self.mask[:,:,:T,:T], torch.finfo(scores.dtype).min)
        pattern = torch.softmax(scores, -1)

        heads = (pattern @ V_h).transpose(1, 2).contiguous()    # contiguous mem
        out = torch.reshape(heads, (B, T, d))
        return self.W_o(out)
    
class AttentionBlock(nn.Module):
    def __init__(self, d: int, H: int, MAXT: int, d_ff: int) -> None:
        super().__init__()
        self.MHA = MultiHeadedAttention(d, H, MAXT)
        self.FFN = FeedForwardNetwork(d, d_ff)
        self.RMS1 = nn.RMSNorm(d)
        self.RMS2 = nn.RMSNorm(d)
    
    def forward(self, X):
        # ... : (B x T x d)
        out = X + self.MHA(self.RMS1(X))
        out = out + self.FFN(self.RMS2(out))
        return out
    
class Transformer(nn.Module):
    # TODO:
    #   dropout
    #   input params
    #   LM head? (bias optional) X
    
    def __init__(self, d: int, L: int, H: int, MAXT: int, V: int, d_ff: int) -> None:
        super().__init__()
        self.d = d
        self.L = L
        self.H = H
        self.MAXT = MAXT
        self.V = V
        self.d_ff = d_ff

        self.embedding = nn.Embedding(V, d)
        self.block = AttentionBlock(d, H, MAXT, d_ff)
        self.RMS = nn.RMSNorm(d)

    def forward(self, X):
        B, T = X.shape
        if T > self.MAXT:
            raise Exception(f'too many tokens; max {self.MAXT} tokens')
        
        out = self.embedding(X)
        out = self.block(out)
        out = self.RMS(out)
        out = out @ self.embedding.weight.T
        return out