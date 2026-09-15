"""inference.py — пример инференса модели best2cc_v5 (разрядные сегменты).

Показывает, как из пакета строится прогноз: конфигурация берётся из
model/config.json, веса — из model/*.pt, данные — списки CellData
(см. coin_cell_pinn.dataset_engine).

Запуск примера на синтетических признаках: python inference.py
(полный прогноз требует реальных циклов; здесь проверяется сборка конвейера).
"""
import json
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from coin_cell_pinn.best2cc import eval_config, tail_grid, rate_stats, cap_exponent
from coin_cell_pinn.population import AgingNet, ResidualNet


def load_model(run_dir: str = "model"):
    cfg = json.load(open(f"{run_dir}/config.json"))
    aging, residual = [], []
    for i in range(2):
        net = AgingNet(5, hidden=48)
        net.load_state_dict(torch.load(f"{run_dir}/aging_net_{i}.pt", map_location="cpu"))
        net.eval()
        aging.append(net)
    n_res = len([f for f in __import__("os").listdir(run_dir)
                 if f.startswith("residual_net_")])
    for i in range(n_res):
        net = ResidualNet(5, hidden=64, layers=3, hist_dim=5)
        net.load_state_dict(torch.load(f"{run_dir}/residual_net_{i}.pt", map_location="cpu"))
        net.eval()
        residual.append(net)
    return cfg, aging, residual


def main():
    cfg, aging, residual = load_model()
    opts = cfg["options"]
    print("pipeline:", cfg["pipeline"])
    print("options :", opts)
    print("метрики контроля:", cfg["test_metrics"])

    # Конфигурация инференса, соответствующая обученной модели
    grid = tail_grid() if opts["soc_grid"] == "tail_refined" else np.linspace(0, 1, 33)
    eval_kwargs = dict(
        q_model="median",
        prefer_ref=False,
        rate_norm=opts["rate_normalized_soh"],
        rc=opts["rc_dynamics"],
    )
    print("eval_config(...):", eval_kwargs, f"soc_grid: {len(grid)} точек")
    print("\nДля полного прогноза подайте список CellData одного разрядного цикла:")
    print("  from coin_cell_pinn.best2cc import eval_config")
    print("  rows = eval_config({cell_id: [CellData, ...]}, comms, aging, residual,")
    print("                     cfg_yaml, grid, collect_curves=True, **eval_kwargs)")


if __name__ == "__main__":
    main()
