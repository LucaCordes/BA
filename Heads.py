import torch 
from torch import nn 

class NormalHead(nn.Module): # aka LinearHead
    def __init__(self, hidden_dim=256, n_const=128):
        super().__init__()
        self.flat = nn.Flatten()
        self.out = nn.Linear(hidden_dim * n_const, 1)
    
    def forward(self, x, padding):
        x = self.flat(x)
        x = self.out(x)
        return x

class MeanPoolHead(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, padding):
        B, C, D = x.shape
        num_const = (padding == 0).sum(1).unsqueeze(-1)  # (B, 1)
        padding = padding.unsqueeze(-1)  # (B, C, 1)
        x = x * (~padding) 
        pooled = x.sum(dim=1) / num_const  # (B, D)
        return self.classifier(pooled)

class MaxPoolHead(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x, padding=None):
        padding = padding.unsqueeze(-1).expand(-1, -1, x.shape[-1]) # (B, C) → (B, C, D)
        x = torch.masked.amax(x, dim=1, mask=~padding) # (B, D)
        return self.linear(x)

class AttentionPoolHead(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True,)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x, padding):
        B, C, D = x.shape
        q = self.query.expand(B, -1, -1)  # (1, 1, D) → (B, 1, D)
        
        attn_out, _ = self.attn(q, x, x, key_padding_mask=padding)  # (B, 1, D)
        pooled = attn_out.squeeze(1)  # (B, 1, D) → (B, D)
        
        return self.linear(pooled)  # (B, 1)
    
class SimpleAttentionPoolHead(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Parameter(torch.randn(self.hidden_dim))
        self.linear = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x, padding):
        attn_scores = x @ self.query  # (B, C)
        attn_weights = torch.masked.softmax(attn_scores, dim=1, mask=~padding).unsqueeze(-1)  # (B, C, 1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (B, D)
        
        y = self.linear(pooled)
        y.attn_weights = attn_weights.detach()
        return y # (B, 1)

class CLSTokenHead(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, num_layers=2, dropout=0):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(hidden_dim))
        self.linear = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, padding):        
        B, C, D = x.shape
        
        cls_token = self.cls_token.expand(B, 1, -1) # (D) → (B, 1, D)
        x = torch.cat([cls_token, x], dim=1) # (B, C, D) → (B, 1 + C, D) 
        
        cls_pad = torch.zeros((B, 1), dtype=torch.bool, device=padding.device)
        padding = torch.cat([cls_pad, padding], dim=1)  # (B, 1 + C)
        
        x = self.transformer(x, src_key_padding_mask=padding)
        
        cls_out = x[:, 0]                              # (B, D)
        logit = self.linear(cls_out) # .squeeze(-1)       # (B, D) → (B, 1) → (B)
        
        return logit

