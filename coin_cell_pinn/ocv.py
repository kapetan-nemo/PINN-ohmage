import torch

EPS = 1e-3


def ocv_from_params(soc: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """OCV(SOC) с лог-сингулярностями (report 2.1).
    k: вектор [K0, K1, K2, K3, K4, A1, A2] (7 коэффициентов).
    """
    s = torch.clamp(soc, EPS, 1.0 - EPS)
    k0, k1, k2, k3, k4 = k[0], k[1], k[2], k[3], k[4]
    a1, a2 = k[5], k[6]
    return (k0 - k1 / s - k2 * s + k3 * torch.log(s) + k4 * torch.log1p(-s)
            + a1 * (2 * s - 1) + a2 * (2 * s - 1) ** 2)


def ocv_default_params(device="cpu") -> torch.Tensor:
    """Стартовые коэффициенты OCV для LIR2025H (NMC/графит, 3.0-4.2 В)."""
    return torch.tensor([3.4, 0.02, 0.10, 0.05, 0.03, 0.30, -0.10],
                        dtype=torch.float32, device=device)
