from torch import nn
import torch
import numpy as np


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        x = (x + 1e-1).log() / 4
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    
class MLP(nn.Module):
    
    def __init__(self, 
                 dim, 
                 device,
                 data_sigma: float = 1.0,
                 num_layers: int = 4,
                 num_hid: int = 256,
                 clip: float = 1e4,
                 *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_hid = num_hid
        self.clip = clip


        self.time_embed = PositionalEmbedding(self.num_hid)

        self.time_coder_state = nn.Sequential(*[
            nn.Linear(self.num_hid, self.num_hid),
            nn.GELU(),
            nn.Linear(self.num_hid, self.num_hid),
        ])
        self.state_time_net = [nn.Sequential(
            *[nn.Linear(self.num_hid+self.dim, self.num_hid), nn.GELU()])] + [nn.Sequential(
            *[nn.Linear(self.num_hid, self.num_hid), nn.GELU()]) for _ in range(self.num_layers-1)] + [
                                                nn.Linear(self.num_hid, self.dim)]
        self.state_time_net = nn.Sequential(*self.state_time_net)
        self.data_sigma = data_sigma

    def _forward(self, input_array, time_array, *args, **kwargs):
        time_array_emb = self.time_embed(time_array)
        t_net1 = self.time_coder_state(time_array_emb)

        extended_input = torch.cat((input_array, t_net1), -1)
        out_state = self.state_time_net(extended_input)
        return torch.clip(out_state, -self.clip, self.clip)
    
    def forward(self, input_array, time_array, *args, **kwargs):
        c_in = 1 / (self.data_sigma**2 + time_array[:, None]**2)**0.5
        out = self._forward(input_array * c_in, time_array, *args, **kwargs)
        c_skip = self.data_sigma**2 / (self.data_sigma**2 + time_array[:, None]**2)
        c_out = self.data_sigma*time_array[:, None] / (self.data_sigma**2 + time_array[:, None]**2)**0.5
        return input_array * c_skip + out * c_out
        
    
