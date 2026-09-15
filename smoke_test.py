"""smoke_test.py — проверка пакета: веса, инференс, монотонность.

Запуск: python smoke_test.py (требуются torch, numpy).
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from coin_cell_pinn.population import AgingNet, ResidualNet


def main():
    # 1. веса
    aging = []
    for i in (0, 1):
        net = AgingNet(5, hidden=48)
        net.load_state_dict(torch.load(f"model/aging_net_{i}.pt", map_location="cpu"))
        net.eval()
        aging.append(net)
    residual = []
    for i in range(5):
        net = ResidualNet(5, hidden=64, layers=3, hist_dim=5)
        net.load_state_dict(torch.load(f"model/residual_net_{i}.pt", map_location="cpu"))
        net.eval()
        residual.append(net)
    print("веса: 2 aging + 5 residual — OK")

    # 2. инференс на синтетической сетке
    static = torch.tensor([[1.2, 0.0025, 0.0025, -0.01, 2.75]], dtype=torch.float32)
    hist = torch.tensor([[0.95, 0.005, 0.01, 0.0, 0.0]], dtype=torch.float32)
    n = torch.tensor([[250.0]], dtype=torch.float32)
    q = float(np.mean([float(a(static, hist, n)[0]) for a in aging]))
    corr = np.mean([
        r(static.expand(33, -1), hist.expand(33, 5),
          torch.full((33, 1), 250.0),
          torch.tensor(np.linspace(0, 1, 33).reshape(-1, 1), dtype=torch.float32)
          ).detach().numpy().ravel() for r in residual], axis=0)
    assert np.isfinite(corr).all() and abs(q - 0.95) < 0.1
    print(f"инференс: q(250) = {q:.3f}, max|ΔV-коррекция| = {np.abs(corr).max()*1000:.1f} мВ — OK")

    # 3. монотонность: SOH не возрастает с номером цикла
    grid = torch.arange(1.0, 400.0)
    feats = torch.zeros(1, 7, dtype=torch.float32)
    for a in aging:
        curve = a(feats, grid)[0].detach().numpy()
        assert (np.diff(curve) <= 1e-6).all()
    print("монотонность SOH-кривой — OK")
    print("SMOKE TEST PASSED")
