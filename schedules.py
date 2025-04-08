import torch

# alpha and alpha bar are inferred from beta
class LinearSchedule:
    def __init__(self, T=100, beta_min=0.0001, beta_max=0.02):
        self.T = T

        self.beta = beta_min + (torch.arange(T) / (T - 1)) * (beta_max - beta_min)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

# alpha and beta are inferred from alpha_bar in the Cosine schedule
class CosineSchedule:
    def __init__(self, T=100, s=0.008):
        self.T = T
        self.s = torch.tensor(s, dtype=torch.float32)
        
        # Calculate initial alpha_bar
        alpha_bar = self._f(torch.arange(T+1)) / self._f(0)
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        
        self.beta = torch.clip(beta, 0.001, 0.999)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def _f(self, t):
        t = torch.tensor(t, dtype=torch.float32)
        return torch.cos(((t / self.T) + self.s) / (1 + self.s) * torch.pi / 2) ** 2