"""training.py — Adam (cosine) → L-BFGS на фиксированной сетке коллокаций (D6)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .dataset_engine import Normalizer, sobol_collocation


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


class RunLogger:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "log.jsonl"

    def log(self, **kw):
        kw["ts"] = time.time()
        with open(self.path, "a") as f:
            f.write(json.dumps(kw) + "\n")

    def save_json(self, name: str, obj):
        with open(self.dir / name, "w") as f:
            json.dump(obj, f, indent=2)


def make_batch(cell, norm: Normalizer, cfg: dict, device: str = "cpu") -> dict:
    """Полный батч по одному разрядному сегменту + сетка коллокаций."""
    t_n = norm.t(cell.t).to(device)
    i_n = norm.i(cell.i).to(device)
    v_n = norm.v(cell.v).to(device)
    soc = cell.soc.to(device)
    batch = {"t_n": t_n, "i_n": i_n, "v_n": v_n, "soc_meas": soc,
             "q_max_ah": cell.q_max_ah}
    if cfg.get("ecm", {}).get("slope_weight", False):
        # равномерная "стоимость" ошибки в SOC-пространстве: w = 1/|dV/dSOC|,
        # нормировано на медиану (средняя точка кривой получает вес 1)
        v_np = cell.v.numpy().ravel()
        s_np = cell.soc.numpy().ravel()
        dv_ds = np.gradient(v_np) / np.maximum(np.abs(np.gradient(s_np)), 1e-4)
        w = 1.0 / (np.abs(dv_ds) / max(np.median(np.abs(dv_ds)), 1e-6) + 0.5)
        batch["w_v"] = torch.tensor((w / np.median(w)).reshape(-1, 1),
                                    dtype=torch.float32)
    # коллокации: Sobol-подвыборка временной оси (фиксируется для L-BFGS)
    n_col = min(cfg["ecm"].get("n_colloc", 512), len(t_n))
    t1 = float(cell.t[-1])
    t_col = torch.tensor(sobol_collocation(n_col, 0.0, t1, cfg["seed"]).reshape(-1, 1),
                         dtype=torch.float32)
    i_phys_col = torch.tensor(np.interp(t_col.ravel(), cell.t.ravel().numpy(), cell.i.ravel().numpy()),
                              dtype=torch.float32).reshape(-1, 1)
    t_col_n = norm.t(t_col)
    i_col_n = norm.i(i_phys_col)
    return {**batch, "t_col_n": t_col_n, "i_col_n": i_col_n, "i_phys_col": i_phys_col}


def fit_ecm_cell(model, cell, norm: Normalizer, cfg: dict, logger: RunLogger | None = None,
                 run_dir: str = "runs/ecm") -> dict:
    from .ecm_pinn_engine import ecm_loss
    device = "cpu"
    batch = make_batch(cell, norm, cfg, device)
    cfg = {**cfg, "_v_max": norm.v_max, "_v_min": norm.v_min}

    hist = []
    # Этап 1: Adam + cosine annealing
    opt = torch.optim.Adam(model.parameters(), lr=cfg["ecm"]["lr0"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["ecm"]["adam_epochs"], eta_min=cfg["ecm"]["lr_min"])
    for epoch in range(cfg["ecm"]["adam_epochs"]):
        opt.zero_grad()
        loss, parts = ecm_loss(model, batch, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if epoch % max(1, cfg["ecm"]["adam_epochs"] // 10) == 0:
            hist.append({"epoch": epoch, **parts, "loss": float(loss)})
            if logger:
                logger.log(stage="adam", epoch=epoch, **parts)

    # Этап 2: L-BFGS на фиксированной сетке (D6)
    if cfg["ecm"].get("lbfgs_iters", 0) > 0:
        opt2 = torch.optim.LBFGS(model.parameters(), max_iter=cfg["ecm"]["lbfgs_iters"],
                                 tolerance_grad=1e-7, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            l, _ = ecm_loss(model, batch, cfg)
            l.backward()
            return l

        opt2.step(closure)

    loss, parts = ecm_loss(model, batch, cfg)
    result = {"cell_id": cell.cell_id, "params": model.params_dict(),
              "ocv_k": model.ocv_k.detach().tolist(),
              "final_loss": parts, "history": hist}
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{run_dir}/{cell.cell_id}.pt")
    if logger:
        logger.save_json(f"ecm_{cell.cell_id}.json", result)
    return result
