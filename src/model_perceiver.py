import torch
import torch.nn as nn
from perceiver_pytorch import PerceiverIO

def fourier_encode_2d(H, W, num_bands=6, max_freq=10.0):
    """
    Returns a tensor of shape (H*W, 4*num_bands+2) for 2D Fourier positional encoding.
    """
    y = torch.linspace(-1., 1., steps=H)
    x = torch.linspace(-1., 1., steps=W)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # (H, W)
    coords = torch.stack([grid_y, grid_x], dim=-1)        # (H, W, 2)
    coords = coords.reshape(-1, 2)                        # (HW, 2)
    # Fourier features
    freq_bands = torch.linspace(1.0, max_freq, num_bands)
    features = [coords]
    for freq in freq_bands:
        for fn in [torch.sin, torch.cos]:
            features.append(fn(coords * freq * torch.pi))
    return torch.cat(features, dim=-1)  # (HW, 2 + 2*num_bands*2)

class PerceiverWrapper(nn.Module):
    def __init__(
        self,
        input_shape=(3, 32, 32),
        latent_dim=512,
        num_latents=128,     
        depth=12,             
        dim=128,            
        logits_dim=128,       
        pos_dim=26            
    ):
        super().__init__()

        C, H, W = input_shape
        self.C = C
        self.H = H
        self.W = W
        self.channel = self.C
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Project image channels to Perceiver input dim
        self.input_proj = nn.Linear(self.C, dim)

        # Output MLP head: PerceiverIO out_dim → C (per-pixel)
        self.output_head = nn.Sequential(
            nn.Linear(logits_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.C)
        )

        self.timestep_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, H * W)
        )

        # 2D positional encoding (HW, pos_dim)
        pe = fourier_encode_2d(H, W, num_bands=(pos_dim-2)//4)
        self.register_buffer('pos_enc', pe, persistent=False)  # Not trainable

        # Perceiver IO
        self.model = PerceiverIO(
            dim=dim + pos_dim,
            queries_dim=dim + pos_dim,
            logits_dim=logits_dim,
            depth=depth,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            weight_tie_layers=False,
        )

        # Learnable query for each pixel (plus pos encoding)
        self.query = nn.Parameter(torch.randn(1, H * W, dim))

    def forward(self, x, t):
        B, C, H, W = x.shape
        device = x.device

        # Project input image to (B, HW, dim)
        x = x.view(B, C, -1).permute(0, 2, 1)           # (B, HW, C)
        t = t.float().view(B, 1) / 1000
        t_embed = self.timestep_mlp(t).view(B, -1, 1)   # (B, HW, 1)
        x = self.input_proj(x + t_embed)                # (B, HW, dim)

        # Add positional encoding to inputs
        pe = self.pos_enc.to(device)                    # (HW, pos_dim)
        pe = pe.unsqueeze(0).expand(B, -1, -1)          # (B, HW, pos_dim)
        x = torch.cat([x, pe], dim=-1)                  # (B, HW, dim+pos_dim)

        # Prepare queries with position info
        q_content = self.query.expand(B, -1, -1)        # (B, HW, dim)
        queries = torch.cat([q_content, pe], dim=-1)    # (B, HW, dim+pos_dim)

        # Run Perceiver IO: output (B, HW, logits_dim)
        out = self.model(x, queries=queries)            # (B, HW, logits_dim)
        out = self.output_head(out)                     # (B, HW, C)
        out = out.view(B, H, W, self.C).permute(0, 3, 1, 2)  # (B, C, H, W)

        return out
