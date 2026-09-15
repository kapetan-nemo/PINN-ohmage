"""population.py — популяционный протокол: 45 ячеек, сплит ~32/13.

Сценарий (рекуррентная регрессия по циклам, финальная версия):
  TRAIN (32 ячейки): для всех доступных циклов строим рекорды
    x = [статические (r0_base, q0, i_nom, early_slope), N, log1p(N), q_prev/q0]
    y = q(N)/q0  — ёмкость следующего цикла
  — «зная предыдущие циклы» = рекуррентный признак q_prev (измеренный).

  TEST (13 held-out ячеек): комиссинг по ранним циклам (ECM-фит → R0),
  далее итерация: OCV-таблица строится из предыдущего ИЗМЕРЕННОГО цикла
  (инверсия ECM), ёмкость q(N) предсказывает сеть, ток I(t) измеренный →
  V(t) ODE-интегратором. Пол схемы (оракул-q) = 8–18 мВ.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .dataset_engine import CoinCellDataPipeline, CellData, Normalizer
from .ocv_emp import build_ocv_table, voltage_pred_from_table

EARLY_CYCLES = [5, 10, 15, 20]
SAMPLE_EVERY = 10
SOC_LO, SOC_HI = 0.45, 0.85   # якоря коррекции кривой


def load_population(cfg: dict, cell_ids: list[int] | list[str] | None = None,
                    source: str = "lir", ioc_dir: str = "data/Dataset_IOC"
                    ) -> dict[str, list[CellData]]:
    """Универсальная загрузка: 'lir' — 45 LIR2025H, 'ioc' — 48 IOC-ячеек."""
    p = CoinCellDataPipeline(cfg)
    p.cfg["data"]["max_points_per_segment"] = 600
    if source == "ioc":
        cells = p.load_ioc(ioc_dir, cell_ids=cell_ids, max_points=600)
    else:
        cycles = sorted({2, *range(SAMPLE_EVERY, 101, SAMPLE_EVERY)})
        cells = p.load_lir2025h(cfg["data"]["lir2025h_zip"], cells_sel=cell_ids,
                                cycles=cycles, max_points=600)
    cells = [c for c in cells if c.q_max_ah > 1e-3]
    out: dict[str, list[CellData]] = {}
    for j, c in enumerate(cells, 1):
        out.setdefault(c.cell_id, []).append(c)
        if j % 500 == 0:
            print(f"  load: {j}/{len(cells)} cycles", flush=True)
    for name in out:
        out[name].sort(key=lambda c: c.cycle_n)
    return out


def split_population(cells_dict: dict[str, list[CellData]], seed: int = 42,
                     fracs=(0.6, 0.2, 0.2)) -> dict[str, list[str]]:
    """Стратифицированный по протоколам сплит (группа = префикс IOC_XXX)."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {}
    for name in sorted(cells_dict):
        key = name.split("_")[1] if name.startswith("IOC_") else "lir"
        groups.setdefault(key, []).append(name)
    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for key, names in sorted(groups.items()):
        names = sorted(names)
        rng.shuffle(names)
        n = len(names)
        n_tr = max(1, int(np.floor(n * fracs[0])))
        n_val = int(np.floor(n * fracs[1])) if n >= 3 else 0
        n_te = max(n - n_tr - n_val, 0) if n >= 3 else 0
        split["train"] += names[:n_tr]
        split["val"] += names[n_tr:n_tr + n_val]
        split["test"] += names[n_tr + n_val:]
    return split


def commission_cell(cells: list[CellData], cfg: dict, run_dir: str | None = None,
                    early_cycles: list[int] | None = None) -> dict:
    """Ранние циклы ячейки → R0 (ECM-фит на одном цикле), Q0, V-окно."""
    from .training import set_seed, fit_ecm_cell
    from .ecm_pinn_engine import Thevenin2RPINNModule
    early = [c for c in cells if c.cycle_n in (early_cycles or EARLY_CYCLES)] or cells[:2]
    norm = Normalizer.fit(early)
    set_seed(cfg["seed"])
    m = Thevenin2RPINNModule(t_ref=norm.t_ref, i_ref=norm.i_ref)
    rd = run_dir or f"runs/population/ecm_{early[0].cell_id}"
    fit_ecm_cell(m, early[0], norm, cfg, run_dir=rd)
    params = m.params_dict()
    q0 = float(np.median([c.q_max_ah for c in early]))
    v_min = float(min(c.v.min() for c in cells))
    return {"params": params, "q0": q0, "i_nom": float(early[0].i.mean()),
            "v_min": v_min, "norm": norm}


def early_slope(cells: list[CellData]) -> float:
    """Ранний наклон деградации: (Q(20)−Q(10))/Q(10) по доступным ранним циклам."""
    q = {c.cycle_n: c.q_max_ah for c in cells}
    if 10 in q and 20 in q and q[10] > 1e-3:
        return (q[20] - q[10]) / q[10]
    qs = sorted(q.items())
    if len(qs) >= 2:
        (n0, qa), (n1, qb) = qs[0], qs[-1]
        return (qb - qa) / max(qa, 1e-6) if n1 > n0 else 0.0
    return 0.0


def build_records(cells: list[CellData], comm: dict) -> list[dict]:
    """Рекорды (cell, N): x=[static, N, log1p(N), q_prev, q_trend]
    → y=[q(N)/q0, ΔV_lo, ΔV_hi] (разности кривых текущий−предыдущий цикл
    в якорях SOC_LO/SOC_HI — быстрые таргеты без интегратора)."""
    q0 = comm["q0"]
    r0_base = comm["params"]["R0"]
    slope = early_slope(cells)

    def soc_at(c: CellData, target: float) -> int:
        return int(np.argmin(np.abs(c.soc.numpy().ravel() - target)))

    def dv(c_cur: CellData, c_prev: CellData, soc_a: float) -> float:
        j_cur = soc_at(c_cur, soc_a)
        j_prev = soc_at(c_prev, soc_a)
        return float(c_cur.v[j_cur] - c_prev.v[j_prev])

    recs = []
    for k, (prev, cur) in enumerate(zip(cells[:-1], cells[1:])):
        prev2 = cells[k - 1] if k > 0 else prev
        trend = (prev.q_max_ah - prev2.q_max_ah) / max(prev.cycle_n - prev2.cycle_n, 1)
        q_lin = prev.q_max_ah / q0 + trend * (cur.cycle_n - prev.cycle_n) / q0
        dlo = dv(cur, prev, SOC_LO); dhi = dv(cur, prev, SOC_HI)
        dlo2 = dv(prev, prev2, SOC_LO); dhi2 = dv(prev, prev2, SOC_HI)
        recs.append({
            "cell": cur.cell_id, "N": cur.cycle_n,
            "static": [r0_base, q0, comm["i_nom"], slope, comm["v_min"]],
            "hist": [q_lin, dlo, dhi, dlo - dlo2, dhi - dhi2],
            "target": [cur.q_max_ah / q0, dlo, dhi],
        })
    return recs


class AgingNet(nn.Module):
    """Рекуррентный регрессор: q(N), ΔV_lo, ΔV_hi."""

    def __init__(self, static_dim: int, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(static_dim + 6, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, static: torch.Tensor, hist: torch.Tensor, n: torch.Tensor):
        x = torch.cat([static, torch.log1p(n.clamp(min=0.0)), hist], dim=1)
        out = self.net(x)
        q = torch.sigmoid(out[:, 0:1]) * 1.5
        dv_lo = torch.tanh(out[:, 1:2]) * 0.15
        dv_hi = torch.tanh(out[:, 2:3]) * 0.15
        return q, dv_lo, dv_hi


def train_aging_net(records: list[dict], cfg: dict, device: str = "cpu") -> AgingNet:
    static_dim = len(records[0]["static"])
    rate_cond = len(records[0]["hist"]) == 4
    X_s = torch.tensor([r["static"] for r in records], dtype=torch.float32, device=device)
    X_h = torch.tensor([r["hist"] for r in records], dtype=torch.float32, device=device)
    X_n = torch.tensor([[r["N"]] for r in records], dtype=torch.float32, device=device)
    Y = torch.tensor([r["target"] for r in records], dtype=torch.float32, device=device)
    net = AgingNet(static_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(cfg["pop"]["adam_epochs"]):
        opt.zero_grad()
        q, dv_lo, dv_hi = net(X_s, X_h, X_n)
        loss = ((q - Y[:, 0:1]) ** 2).mean() \
               + ((dv_lo - Y[:, 1:2]) ** 2).mean() + ((dv_hi - Y[:, 2:3]) ** 2).mean()
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def eval_population(test_cells: dict[str, list[CellData]], comms: dict[str, dict],
                    nets, cfg: dict, q_pow_w: float = 0.0,
                    anchor_mode: str = "net", n_early: int = 20,
                    res_nets=None, soc_grid: np.ndarray | None = None,
                    ocv_avg_k: int = 1, clean_filter: bool = False,
                    rate_match: bool = False, rate_cond: bool = False,
                    res_extra: bool = False, ocv_tau: float | None = None,
                    q_est_feat: bool = False) -> list[dict]:
    from .validate import rmse, mae, monotonicity_check

    def soc_at(c: CellData, target: float) -> int:
        return int(np.argmin(np.abs(c.soc.numpy().ravel() - target)))

    def dv(c_cur: CellData, c_prev: CellData, soc_a: float) -> float:
        return float(c_cur.v[soc_at(c_cur, soc_a)] - c_prev.v[soc_at(c_prev, soc_a)])

    nets = nets if isinstance(nets, (list, tuple)) else [nets]
    q_pow_w = float(q_pow_w)
    rate_match = bool(rate_match)
    rate_cond = bool(rate_cond)

    def predict_q_pow(hist_pairs, n_target, q0):
        """Робастная экстраполяция ёмкости: медиана наклонов по последним 6
        парам измеренных циклов (LSQ-прямая чувствительна к выбросам формации)."""
        if len(hist_pairs) < 2:
            return hist_pairs[-1][1] / q0 if hist_pairs else 1.0
        pts = hist_pairs[-6:]
        slopes = []
        for (n1, q1), (n2, q2) in zip(pts[:-1], pts[1:]):
            if n2 > n1:
                slopes.append((q2_ := q2) if False else (q2 - q1) / (n2 - n1))
        slope = float(np.median(slopes)) if slopes else 0.0
        q = pts[-1][1] + slope * (n_target - pts[-1][0])
        return float(np.clip(q, 0.05 * q0, 1.2 * q0)) / q0

    rows = []
    n_tot = len(test_cells)
    for j, (b, cells) in enumerate(test_cells.items(), 1):
        print(f"  eval [{j}/{n_tot}] {b}...", flush=True)
        comm = comms[b]
        q0, r0_base = comm["q0"], comm["params"]["R0"]
        sim = {"R0": r0_base, "R1": 1e-6, "C1": 1e6, "R2": 1e-6, "C2": 1e6}
        slope = early_slope(cells)
        early = [c for c in cells if c.cycle_n <= n_early]
        qs = [c.q_max_ah for c in early]
        ns = [c.cycle_n for c in early]
        trend = (qs[-1] - qs[-2]) / max(ns[-1] - ns[-2], 1) if len(qs) > 1 else 0.0
        hist_lo = dv(early[-1], early[-2], SOC_LO) if len(early) > 1 else 0.0
        hist_hi = dv(early[-1], early[-2], SOC_HI) if len(early) > 1 else 0.0
        soh_true, soh_pred, rmses, rmses_or, rmses_clean = [], [], [], [], []
        hist_pairs = [(c.cycle_n, c.q_max_ah) for c in early]
        meas_hist = [(c.cycle_n, float(c.i.mean()), c) for c in early]  # (N, I, cell)
        ah_cum_run = float(sum(c.q_max_ah for c in early))
        q_est = 1.0
        dlo2 = dhi2 = 0.0
        j_ = next((k for k, c in enumerate(cells) if c.cycle_n > n_early), len(cells))
        for prev, cur in zip(cells[:-1], cells[1:]):
            if prev.cycle_n <= n_early:
                continue
            j_ += 1
            static = torch.tensor([r0_base, q0, comm["i_nom"], slope, comm["v_min"]],
                                  dtype=torch.float32).unsqueeze(0)
            i_cur = float(cur.i.mean())
            q_lin = prev.q_max_ah / q0 + trend * (cur.cycle_n - prev.cycle_n) / q0
            dlo = hist_lo; dhi = hist_hi
            h = torch.tensor([[q_lin, dlo, dhi, dlo - dlo2, dhi - dhi2]], dtype=torch.float32)
            n = torch.tensor([[float(cur.cycle_n)]], dtype=torch.float32)
            outs = [nt(static, h, n) for nt in nets]
            q_n = float(np.mean([float(o[0]) for o in outs]))
            if anchor_mode == "persistence":
                dv_lo, dv_hi = dlo, dhi
            else:
                dv_lo = float(np.mean([float(o[1]) for o in outs]))
                dv_hi = float(np.mean([float(o[2]) for o in outs]))
            if q_pow_w > 0.0:
                q_pow = predict_q_pow(hist_pairs, cur.cycle_n, q0)
                q_n = (1.0 - q_pow_w) * q_n + q_pow_w * q_pow
            if rate_match:
                # ближайшие по току измеренные циклы (окно ±20 %, иначе все)
                i_cur = float(cur.i.mean())
                cands = [(abs(im - i_cur), n_, c_) for n_, im, c_ in meas_hist
                         if abs(im / max(i_cur, 1e-6) - 1.0) < 0.2] or \
                       sorted([(abs(im - i_cur), n_, c_) for n_, im, c_ in meas_hist])[:1]
                past = [c_ for _, n_, c_ in sorted(cands)[:ocv_avg_k]]
                table_prev = build_ocv_table(past, r0_base)
            elif ocv_avg_k > 1:
                past = [c for c in cells[:j_] if c.cycle_n <= prev.cycle_n][-ocv_avg_k:]
                if ocv_tau:
                    tables = [build_ocv_table([c_], r0_base)[1] for c_ in past]
                    wts = np.exp(np.array([-(past[-1].cycle_n - p.cycle_n) / ocv_tau
                                           for p in past]))
                    wts = wts / wts.sum()
                    soc_g = build_ocv_table([past[-1]], r0_base)[0]
                    v_g = np.average(np.vstack(tables), axis=0, weights=wts)
                    table_prev = (soc_g, v_g)
                else:
                    table_prev = build_ocv_table(past, r0_base)
            else:
                table_prev = build_ocv_table([prev], r0_base)
            soc_g, v_g = table_prev
            if res_nets is not None and soc_grid is not None:
                # остаточная коррекция ΔV(SOC): усреднение по ансамблю сетей
                n_grid = len(soc_grid)
                hist_row = [q_lin, hist_lo, hist_hi]
                if rate_cond:
                    hist_row += [i_cur / max(comm["i_nom"], 1e-6)]
                if res_extra:
                    hist_row += [ah_cum_run / max(comm["q0"], 1e-6),
                                 (float(cur.meta.get("T_mean", 23.0)) - 23.0) / 5.0]
                if q_est_feat:
                    hist_row += [q_est]
                h_c = torch.tensor([hist_row], dtype=torch.float32).expand(n_grid, len(hist_row))
                n_c = torch.full((n_grid, 1), float(cur.cycle_n))
                static_c = static.expand(n_grid, -1)
                soc_t = torch.tensor(soc_grid.reshape(-1, 1), dtype=torch.float32)
                corr_grid = np.mean(
                    [nt(static_c, h_c, n_c, soc_t).detach().numpy().ravel() for nt in res_nets],
                    axis=0)
                corr = np.interp(soc_g, soc_grid, corr_grid)
            else:
                corr = float(dv_lo) + (float(dv_hi) - float(dv_lo)) * np.clip(
                    (soc_g - SOC_LO) / (SOC_HI - SOC_LO), 0.0, 1.0)
            table_n = (soc_g, v_g + corr)
            vp = voltage_pred_from_table(sim, table_n, cur.t.numpy(),
                                         cur.i.numpy(), q0 * q_n)
            t_np = cur.t.numpy().ravel()
            i_np = cur.i.numpy().ravel()
            q_coul = float(np.trapezoid(i_np, t_np) / 3600.0) if hasattr(np, "trapezoid") \
                else float(np.trapz(i_np, t_np) / 3600.0)
            vp_or = voltage_pred_from_table(sim, table_n, cur.t.numpy(),
                                            cur.i.numpy(), q_coul)
            rmses.append(rmse(vp, cur.v.numpy().ravel()))
            rmses_or.append(rmse(vp_or, cur.v.numpy().ravel()))
            if clean_filter:
                im = float(np.median([x.i.median().item() for x in cells])) if len(cells) > 3 else float(cur.i.mean())
                dur = cur.cycle_n and (cur.t.numpy().ravel()[-1])
                is_clean = (0.9 * im <= cur.i.median() <= 1.1 * im
                            and cur.soc.min() < 0.03)
                if is_clean:
                    rmses_clean.append(rmse(vp, cur.v.numpy().ravel()))
            soh_true.append(cur.q_max_ah / q0)
            soh_pred.append(q_n)
            trend = (cur.q_max_ah - prev.q_max_ah) / max(cur.cycle_n - prev.cycle_n, 1)
            dlo2, dhi2 = dlo, dhi
            hist_lo = dv(cur, prev, SOC_LO)
            hist_hi = dv(cur, prev, SOC_HI)
            if (not rate_match) or abs(i_cur / max(comm["i_nom"], 1e-6) - 1.0) < 0.15:
                hist_pairs.append((cur.cycle_n, cur.q_max_ah))
            meas_hist.append((cur.cycle_n, i_cur, cur))
            ah_cum_run += cur.q_max_ah
            q_est = hist_pairs[-1][1] / max(comm["q0"], 1e-6)
        if rmses:
            rows.append({
                "cell": b,
                "rmse_v_mean_mv": round(1000 * float(np.mean(rmses)), 1),
                "rmse_v_max_mv": round(1000 * float(np.max(rmses)), 1),
                "rmse_v_coulomb_mv": round(1000 * float(np.mean(rmses_or)), 1),
                "rmse_v_clean_mv": round(1000 * float(np.mean(rmses_clean)), 1) if rmses_clean else None,
                "n_clean": len(rmses_clean),
                "soh_mae_pp": round(100 * mae(soh_pred, soh_true), 2),
                "soh_monotone": monotonicity_check(soh_pred, tol=2e-2),
                "n_test_cycles": len(rmses),
            })
    return rows


def build_residual_records(cells: list[CellData], comm: dict, soc_grid: np.ndarray,
                           soc_step: int = 1) -> list[dict]:
    """Рекорды остаточной коррекции: ΔV(SOC) между последовательными циклами.

    Для каждой пары (prev, cur) — вектор ΔV на сетке soc_grid; признаки сети
    те же, что у AgingNet (статика комиссинга + N + q_lin + якоря из истории).
    """
    q0 = comm["q0"]
    r0_base = comm["params"]["R0"]
    slope = early_slope(cells)

    def soc_curve(c: CellData) -> tuple[np.ndarray, np.ndarray]:
        s = c.soc.numpy().ravel()
        v = c.v.numpy().ravel()
        order = np.argsort(s)
        return s[order], v[order]

    rate_cond = comm.get("rate_cond", False)
    extra_feats = bool(comm.get("res_extra", False))
    ah_cum = 0.0
    recs = []
    for k, (prev, cur) in enumerate(zip(cells[:-1], cells[1:])):
        prev2 = cells[k - 1] if k > 0 else prev
        trend = (prev.q_max_ah - prev2.q_max_ah) / max(prev.cycle_n - prev2.cycle_n, 1)
        q_lin = prev.q_max_ah / q0 + trend * (cur.cycle_n - prev.cycle_n) / q0
        sp, vp_ = soc_curve(prev)
        sc, vc = soc_curve(cur)
        dvds_prev = np.abs(np.gradient(vp_) / np.maximum(np.abs(np.gradient(sp)), 1e-4))
        c_ref = max(float(np.median(dvds_prev)), 1e-6)
        dv_lo = float(np.interp(SOC_LO, sc, vc) - np.interp(SOC_LO, sp, vp_))
        dv_hi = float(np.interp(SOC_HI, sc, vc) - np.interp(SOC_HI, sp, vp_))
        rate = float(cur.i.mean()) / max(comm["i_nom"], 1e-6)
        ah_cum += prev.q_max_ah
        extra = []
        if extra_feats:
            extra = [ah_cum / q0,
                     (float(cur.meta.get("T_mean", 23.0)) - 23.0) / 5.0]
        for soc_j in soc_grid[::soc_step]:
            rec = {
                "cell": cur.cell_id, "N": cur.cycle_n, "soc": float(soc_j),
                "static": [r0_base, q0, comm["i_nom"], slope, comm["v_min"]],
                "hist": [q_lin, dv_lo, dv_hi],
                "target": float(np.interp(soc_j, sc, vc) - np.interp(soc_j, sp, vp_)),
            }
            if rate_cond:
                rec["hist"] = rec["hist"] + [rate]
            if extra_feats:
                rec["hist"] = rec["hist"] + extra
            # равная «стоимость» ошибки в координате SOC
            dvds_j = float(np.interp(soc_j, sp, dvds_prev))
            rec["weight"] = 1.0 / (dvds_j / c_ref + 0.5)
            recs.append(rec)
    return recs


class ResidualNet(nn.Module):
    """Сеть остаточной коррекции ΔV(SOC; признаки ячейки, номер цикла)."""

    def __init__(self, static_dim: int, hidden: int = 64, layers: int = 3,
                 rate_cond: bool = False, hist_dim: int | None = None):
        super().__init__()
        self.rate_cond = rate_cond
        if hist_dim is None:
            hist_dim = 4 if rate_cond else 3
        # вход: static(5) + [soc, log1p(N), hist...]
        net: list[nn.Module] = [nn.Linear(static_dim + 2 + hist_dim, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.SiLU()]
        net += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*net)

    def forward(self, static: torch.Tensor, hist: torch.Tensor, n: torch.Tensor,
                soc: torch.Tensor):
        x = torch.cat([static, soc, torch.log1p(n.clamp(min=0.0)), hist], dim=1)
        return torch.tanh(self.net(x)) * 0.15


def train_residual_net(records: list[dict], cfg: dict, device: str = "cpu",
                       seed: int = 42, max_samples: int = 200000,
                       hidden: int = 64, layers: int = 3,
                       lbfgs_polish: int = 0, lr: float = 2e-3) -> ResidualNet:
    from .training import set_seed
    set_seed(seed)
    torch.manual_seed(seed)
    static_dim = len(records[0]["static"])
    hist_dim = len(records[0]["hist"])
    X_s = torch.tensor([r["static"] for r in records], dtype=torch.float32, device=device)
    X_h = torch.tensor([r["hist"] for r in records], dtype=torch.float32, device=device)
    X_n = torch.tensor([[r["N"]] for r in records], dtype=torch.float32, device=device)
    X_c = torch.tensor([[r["soc"]] for r in records], dtype=torch.float32, device=device)
    Y = torch.tensor([[r["target"]] for r in records], dtype=torch.float32, device=device)
    W = torch.tensor([[r.get("weight", 1.0)] for r in records], dtype=torch.float32, device=device)
    if len(records) > max_samples:
        idx = torch.randperm(len(records))[:max_samples]
        X_s, X_h, X_n, X_c, Y, W = X_s[idx], X_h[idx], X_n[idx], X_c[idx], Y[idx], W[idx]
    net = ResidualNet(static_dim, hidden=hidden, layers=layers,
                      hist_dim=hist_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bs = min(8192, len(X_s))
    for _ in range(cfg["pop"]["adam_epochs"]):
        perm = torch.randperm(len(X_s), device=device)
        for i in range(0, len(X_s), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = net(X_s[idx], X_h[idx], X_n[idx], X_c[idx])
            loss = (W[idx] * (pred - Y[idx]) ** 2).mean()
            loss.backward()
            opt.step()
    if lbfgs_polish > 0:
        opt2 = torch.optim.LBFGS(net.parameters(), max_iter=lbfgs_polish,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            pred = net(X_s, X_h, X_n, X_c)
            loss = (W * (pred - Y) ** 2).mean()
            loss.backward()
            return loss

        opt2.step(closure)
    return net
