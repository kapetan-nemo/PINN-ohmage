"""best2cc.py — общие утилиты серии экспериментов best2_cc.

Содержит: определение референсных (низкотоковых) циклов, скоростную нормировку
ёмкости, построение OCV-таблицы с опорой на референсные циклы, модели
экстраполяции ёмкости и единый контур оценки конфигураций.
"""
from __future__ import annotations

import numpy as np
import torch

from .ocv_emp import build_ocv_table, voltage_pred_from_table
from .population import early_slope, SOC_LO, SOC_HI


def rate_stats(cells, comm):
    """Медиана рабочего тока и список референсных (низкотоковых) циклов."""
    rates = np.array([float(c.i.median()) for c in cells])
    i_work = float(np.median(rates))
    ref_mask = rates < 0.5 * i_work
    ref = [c for c, m in zip(cells, ref_mask) if m]
    return {"i_work": i_work, "n_ref": len(ref), "ref_cycles": [c.cycle_n for c in ref]}


def ref_factors(cells, comm, half=4):
    """Кратности ёмкости референсных (низкотоковых) циклов относительно локального
    тренда обычных циклов. Возвращает список (cycle_n, ratio) по наблюдениям."""
    st = rate_stats(cells, comm)
    ref_set = set(st["ref_cycles"])
    out = []
    for c in cells:
        if c.cycle_n not in ref_set:
            continue
        neigh = [x.q_max_ah for x in cells
                 if abs(x.cycle_n - c.cycle_n) <= half and x.cycle_n not in ref_set]
        if len(neigh) >= 2:
            out.append((c.cycle_n, float(c.q_max_ah) / float(np.median(neigh))))
    return out


def ref_factor_estimate(factors, upto_n, last=3):
    """Оценка кратности по наблюдениям до цикла upto_n (медиана последних)."""
    vals = [r for n, r in factors if n < upto_n]
    if not vals:
        return 1.0
    return float(np.median(vals[-last:]))


def cap_exponent(cells, comm):
    """Показатель скоростной зависимости ёмкости: q_ref/q_work = (I_work/I_ref)^k."""
    st = rate_stats(cells, comm)
    if st["n_ref"] == 0:
        return 0.0
    ref = [c for c in cells if c.cycle_n in set(st["ref_cycles"])]
    work = [c for c in cells if c.cycle_n not in set(st["ref_cycles"])]
    if not ref or not work:
        return 0.0
    q_ref = float(np.median([c.q_max_ah for c in ref]))
    q_w = float(np.median([c.q_max_ah for c in work]))
    i_ref = float(np.median([c.i.median() for c in ref]))
    if q_w <= 0 or i_ref <= 0:
        return 0.0
    ratio_q = q_ref / q_w
    ratio_i = st["i_work"] / i_ref
    if ratio_q <= 0 or ratio_i <= 0 or abs(np.log(ratio_i)) < 1e-6:
        return 0.0
    k = float(np.log(ratio_q) / np.log(ratio_i))
    return float(np.clip(k, 0.0, 0.5))


def ocv_table_rate(cells, prev, comm, r0, k_avg=3, tau=5.0, prefer_ref=True):
    """OCV-таблица: опора на референсные циклы, иначе свежевзвешенное среднее.

    Возвращает (soc_grid, v_grid, i_table) — средний ток циклов, по которым
    построена таблица (нужен для снятия поглощённой поляризации).
    """
    st = rate_stats(cells, comm)
    ref_set = set(st["ref_cycles"])
    if prefer_ref and st["n_ref"] > 0:
        past_ref = [c for c in cells if c.cycle_n <= prev.cycle_n and c.cycle_n in ref_set]
        if past_ref:
            chosen = past_ref[-min(k_avg, len(past_ref)):]
            soc_g, v_g = build_ocv_table(chosen, r0)
            i_tab = float(np.mean([float(c_.i.mean()) for c_ in chosen]))
            return soc_g, v_g, i_tab
    past = [c for c in cells if c.cycle_n <= prev.cycle_n][-k_avg:]
    soc_g = build_ocv_table([past[-1]], r0)[0]
    tables = np.vstack([build_ocv_table([c_], r0)[1] for c_ in past])
    if tau and len(past) > 1:
        wts = np.exp(np.array([-(past[-1].cycle_n - p.cycle_n) / tau for p in past]))
        wts /= wts.sum()
        v_g = np.average(tables, axis=0, weights=wts)
    else:
        v_g = tables.mean(axis=0)
    i_tab = float(np.mean([float(c_.i.mean()) for c_ in past]))
    return soc_g, v_g, i_tab


def predict_q_median(hist_pairs, n_target, q0, window=6):
    """Медианный наклон последних измеренных циклов (базовая модель)."""
    if len(hist_pairs) < 2:
        return hist_pairs[-1][1] / q0 if hist_pairs else 1.0
    pts = hist_pairs[-window:]
    slopes = [(q2 - q1) / (n2 - n1) for (n1, q1), (n2, q2) in zip(pts[:-1], pts[1:])
              if n2 > n1]
    sl = float(np.median(slopes)) if slopes else 0.0
    q = pts[-1][1] + sl * (n_target - pts[-1][0])
    return float(np.clip(q, 0.05 * q0, 1.2 * q0)) / q0


def predict_q_powerlaw(hist_pairs, n_target, q0, window=12):
    """Взвешенная подгонка q(n) = A - B*sqrt(n) по последним измеренным циклам."""
    pts = hist_pairs[-window:]
    if len(pts) < 3:
        return predict_q_median(hist_pairs, n_target, q0)
    ns = np.array([p[0] for p in pts], float)
    qs = np.array([p[1] for p in pts], float)
    x = np.sqrt(ns)
    w = np.exp(-(ns[-1] - ns) / max(ns[-1] - ns[0], 1) * 3.0)
    Aw = np.vstack([np.ones_like(x), -x]).T * w[:, None]
    coef, *_ = np.linalg.lstsq(Aw, qs * w, rcond=None)
    q = coef[0] - coef[1] * np.sqrt(n_target)
    return float(np.clip(q, 0.05 * q0, 1.2 * q0)) / q0


def tail_grid(n_lo=13, n_hi=21):
    """Сетка SOC с повышенной плотностью у нуля (участок отсечки)."""
    lo = np.linspace(0.0, 0.2, n_lo, endpoint=False)
    hi = np.linspace(0.2, 1.0, n_hi)
    return np.concatenate([lo, hi])


def eval_config(cells_dict, comms, aging_nets, res_nets, cfg, soc_grid,
                q_model="median", prefer_ref=False, rate_norm=False,
                n_early=20, ocv_avg_k=3, ocv_tau=5.0, res_extra=True,
                collect_curves=False, rc=False, rate_norm_features=True,
                ref_factor=False):
    """Единый контур оценки: прогноз V(t) и SOH с опциями серии best2_cc."""
    from .validate import rmse, mae, monotonicity_check

    rows = []
    for name, cells in cells_dict.items():
        comm = comms[name]
        q0, r0_base = comm["q0"], comm["params"]["R0"]
        k = cap_exponent(cells, comm) if (rate_norm or prefer_ref) else 0.0
        st_r = rate_stats(cells, comm)
        i_work = st_r["i_work"]
        ref_set = set(st_r["ref_cycles"])
        factors = ref_factors(cells, comm) if ref_factor else []

        def q_norm(q_ah, i_med, cycle_n=None):
            if ref_factor:
                f = ref_factor_estimate(factors, cycle_n) if cycle_n in ref_set else 1.0
                return q_ah / f
            return q_ah * (i_med / i_work) ** k if (rate_norm and i_med > 0) else q_ah

        prm = comm["params"]
        if rc:
            sim = {"R0": r0_base, "R1": prm["R1"], "C1": prm["C1"],
                   "R2": prm["R2"], "C2": prm["C2"]}
        else:
            sim = {"R0": r0_base, "R1": 1e-6, "C1": 1e6, "R2": 1e-6, "C2": 1e6}
        slope = early_slope(cells)
        early = [c for c in cells if c.cycle_n <= n_early]
        if len(early) < 2:
            continue
        norm_feat = rate_norm and rate_norm_features
        hist_pairs = [(c.cycle_n, q_norm(c.q_max_ah, float(c.i.median()), c.cycle_n)
                       if (norm_feat or ref_factor) else c.q_max_ah) for c in early]
        trend = (hist_pairs[-1][1] - hist_pairs[-2][1]) / max(
            hist_pairs[-1][0] - hist_pairs[-2][0], 1)
        static0 = [r0_base, q0, comm["i_nom"], slope, comm["v_min"]]
        ah_cum = sum(c.q_max_ah for c in early)

        def dv(c1, c0, a):
            j1 = int(np.argmin(np.abs(c1.soc.numpy().ravel() - a)))
            j0 = int(np.argmin(np.abs(c0.soc.numpy().ravel() - a)))
            return float(c1.v[j1] - c0.v[j0])

        hist_lo = dv(early[-1], early[-2], SOC_LO)
        hist_hi = dv(early[-1], early[-2], SOC_HI)
        dlo2 = dhi2 = 0.0
        soh_true, soh_pred, rmses = [], [], []
        curves = []
        for prev, cur in zip(cells[:-1], cells[1:]):
            if prev.cycle_n <= n_early:
                continue
            q_n = (predict_q_powerlaw(hist_pairs, cur.cycle_n, q0)
                   if q_model == "powerlaw"
                   else predict_q_median(hist_pairs, cur.cycle_n, q0))
            static = torch.tensor(static0, dtype=torch.float32).unsqueeze(0)
            q_lin = hist_pairs[-1][1] / q0 + trend * (cur.cycle_n - prev.cycle_n) / q0
            dlo, dhi = hist_lo, hist_hi
            row = [q_lin, dlo, dhi]
            if res_extra:
                row += [ah_cum / q0,
                        (float(cur.meta.get("T_mean", 23.0)) - 23.0) / 5.0]
            n_grid = len(soc_grid)
            h_g = torch.tensor([row], dtype=torch.float32).expand(n_grid, len(row))
            static_g = static.expand(n_grid, -1)
            n_ = torch.full((n_grid, 1), float(cur.cycle_n))
            soc_t = torch.tensor(soc_grid.reshape(-1, 1), dtype=torch.float32)
            corr_grid = np.mean([r(static_g, h_g, n_, soc_t).detach().numpy().ravel()
                                 for r in res_nets], axis=0)
            soc_g, v_g, i_tab = ocv_table_rate(cells, prev, comm, r0_base,
                                               k_avg=ocv_avg_k, tau=ocv_tau,
                                               prefer_ref=prefer_ref)
            corr = np.interp(soc_g, soc_grid, corr_grid)
            v1_0 = v2_0 = 0.0
            if rc:
                v_g = v_g + i_tab * (sim["R1"] + sim["R2"])
                i_cur0 = float(cur.i.mean())
                v1_0, v2_0 = i_cur0 * sim["R1"], i_cur0 * sim["R2"]
            vp = voltage_pred_from_table(sim, (soc_g, v_g + corr), cur.t.numpy(),
                                         cur.i.numpy(), q0 * q_n,
                                         v1_0=v1_0, v2_0=v2_0)
            v_true = cur.v.numpy().ravel()
            rmses.append(rmse(vp, v_true))
            soh_true.append(q_norm(cur.q_max_ah, float(cur.i.median()), cur.cycle_n) / q0)
            if ref_factor and cur.cycle_n in ref_set:
                q_n = q_n * ref_factor_estimate(factors, cur.cycle_n)
            soh_pred.append(q_n)
            if collect_curves:
                curves.append({"N": int(cur.cycle_n), "soh_true": soh_true[-1],
                               "soh_pred": q_n,
                               "rmse_mv": 1000 * float(np.sqrt(np.mean((vp - v_true) ** 2))),
                               "t_h": (cur.t.numpy().ravel() / 3600.0).tolist(),
                               "v_pred": vp.tolist(), "v_true": v_true.tolist()})
            trend = (q_norm(cur.q_max_ah, float(cur.i.median()))
                     - q_norm(prev.q_max_ah, float(prev.i.median()))) / max(
                cur.cycle_n - prev.cycle_n, 1)
            dlo2, dhi2 = dlo, dhi
            hist_lo, hist_hi = dv(cur, prev, SOC_LO), dv(cur, prev, SOC_HI)
            hist_pairs.append((cur.cycle_n,
                               q_norm(cur.q_max_ah, float(cur.i.median()), cur.cycle_n)
                               if (norm_feat or ref_factor) else cur.q_max_ah))
            ah_cum += cur.q_max_ah
        if rmses:
            row = {"cell": name,
                   "rmse_v_mean_mv": round(1000 * float(np.mean(rmses)), 1),
                   "rmse_v_max_mv": round(1000 * float(np.max(rmses)), 1),
                   "soh_mae_pp": round(100 * mae(soh_pred, soh_true), 2),
                   "soh_monotone": monotonicity_check(soh_pred, tol=2e-2),
                   "n_test_cycles": len(rmses), "rate_exponent": round(k, 3)}
            if collect_curves:
                row["curves"] = curves
            rows.append(row)
    return rows
