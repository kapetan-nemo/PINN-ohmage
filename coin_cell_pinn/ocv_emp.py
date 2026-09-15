"""ocv_emp.py — эмпирическая OCV(SOC) из ранних циклов (инверсия ECM).

Идея: на гальваностатическом разряде OCV(soc) ≈ V_meas(soc) + R0·I
(поляризационные V1/V2 медленно релаксируют, при 0.2C их вклад мал и
частично постоянен). OCV-кривая после приработки инвариантна по циклам,
если SOC нормирован на ёмкость текущего цикла.
"""
from __future__ import annotations

import numpy as np
import torch

from .dataset_engine import CellData


def build_ocv_table(cells: list[CellData], r0: float, soc_grid: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Усредняет V(soc)+R0·I по ранним циклам → (soc_grid, ocv_grid).
    Сетка покрывает [0, 1] целиком, включая обрыв V на cutoff (SOC→0)."""
    if soc_grid is None:
        soc_grid = np.linspace(0.0, 1.0, 201)
    curves = []
    for c in cells:
        soc = c.soc.numpy().ravel()
        v = c.v.numpy().ravel() + r0 * c.i.numpy().ravel()
        order = np.argsort(soc)
        curves.append(np.interp(soc_grid, soc[order], v[order]))
    return soc_grid, np.mean(curves, axis=0)


def ocv_from_table(table: tuple[np.ndarray, np.ndarray], soc: float) -> float:
    soc_g, v_g = table
    return float(np.interp(np.clip(soc, soc_g[0], soc_g[-1]), soc_g, v_g))


@torch.no_grad()
def voltage_pred_from_table(params: dict, ocv_table, t_s, i_a, q_max_ah,
                            eta_c: float = 1.0, soc0: float = 1.0,
                            dt_int: float = 1.0) -> np.ndarray:
    """Прогноз V(t) на произвольном профиле I(t): интегрирование 2RC + OCV-таблица.

    τ-устойчивость: внутренний шаг h_int <= tau_min/5.
    """
    R0, R1, C1 = params["R0"], params["R1"], params["C1"]
    R2, C2 = params["R2"], params["C2"]
    t = np.asarray(t_s, float).ravel()
    i = np.asarray(i_a, float).ravel()
    tau1, tau2 = R1 * C1, R2 * C2
    h_int = min(dt_int, min(tau1, tau2) / 5.0) if min(tau1, tau2) > 0 else dt_int

    v1 = v2 = 0.0
    soc = float(soc0)
    out = np.empty(len(t))
    out[0] = ocv_from_table(ocv_table, soc) - R0 * i[0]
    for j in range(1, len(t)):
        t_prev, t_cur = t[j - 1], t[j]
        i_prev, i_cur = i[j - 1], i[j]
        n_sub = max(int(np.ceil((t_cur - t_prev) / h_int)), 1)
        h = (t_cur - t_prev) / n_sub
        for s in range(1, n_sub + 1):
            I = i_prev + (i_cur - i_prev) * (s - 0.5) / n_sub
            e1, e2 = np.exp(-h / tau1), np.exp(-h / tau2)
            v1 = v1 * e1 + I * R1 * (1 - e1)
            v2 = v2 * e2 + I * R2 * (1 - e2)
            soc = max(soc - eta_c * I * h / (3600.0 * q_max_ah), 0.0)
        out[j] = ocv_from_table(ocv_table, soc) - R0 * i_cur - v1 - v2
    return out
