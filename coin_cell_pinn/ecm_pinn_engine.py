"""ecm_pinn_engine.py — Thevenin 2RC PINN (report 4.1, ревизии D2/D3/D4/D12).

Сеть: (t̃, Ĩ) → (V1, V2, SOC); V_t = OCV(SOC;K) − R0·I − V1 − V2 (аналитически).
Параметры R0,R1,C1,R2,C2,K — дифференцируемые, лог-параметризация.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ocv import ocv_from_params, ocv_default_params, EPS

LOG_R0_0, LOG_R1_0, LOG_C1_0, LOG_R2_0, LOG_C2_0 = 0.0, 0.0, 0.0, 0.0, 0.0


class Thevenin2RPINNModule(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3, t_ref: float = 3600.0,
                 i_ref: float = 0.2, device: str = "cpu"):
        super().__init__()
        net: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*net)

        self.log_R0 = nn.Parameter(torch.tensor(LOG_R0_0))
        self.log_R1 = nn.Parameter(torch.tensor(LOG_R1_0))
        self.log_C1 = nn.Parameter(torch.tensor(LOG_C1_0))
        self.log_R2 = nn.Parameter(torch.tensor(LOG_R2_0))
        self.log_C2 = nn.Parameter(torch.tensor(LOG_C2_0))
        self.ocv_k = nn.Parameter(ocv_default_params(device))

        self.t_ref, self.i_ref = t_ref, i_ref

    @property
    def R0(self): return torch.exp(self.log_R0)
    @property
    def R1(self): return torch.exp(self.log_R1)
    @property
    def C1(self): return torch.exp(self.log_C1)
    @property
    def R2(self): return torch.exp(self.log_R2)
    @property
    def C2(self): return torch.exp(self.log_C2)

    def forward(self, t, i):
        """t, i — нормированные (t̃, Ĩ). Возвращает v_term, v1, v2, soc (нормированные V)."""
        out = self.net(torch.cat([t, i], dim=1))
        # выходы сети — малые добавки; базовые масштабы задаёт OCV/IR
        v1 = 0.5 * out[:, 0:1]
        v2 = 0.5 * out[:, 1:2]
        soc = torch.clamp(torch.sigmoid(out[:, 2:3]), EPS, 1.0 - EPS)
        v_term = ocv_from_params(soc, self.ocv_k) - self.R0 * i - v1 - v2
        return v_term, v1, v2, soc

    def params_dict(self) -> dict:
        return {k: float(v) for k, v in dict(
            R0=self.R0, R1=self.R1, C1=self.C1, R2=self.R2, C2=self.C2).items()}


def ecm_loss(model: Thevenin2RPINNModule, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: t_n, i_n, v_n, soc_meas (data-точки) + t_col_n, i_col_n, i_phys_col (коллокации)."""
    lam = cfg["ecm"]
    scale_v = (cfg["_v_max"] - cfg["_v_min"])

    # --- data-часть (опционально: вес по локальному наклону кривой V(SOC) —
    # единая "стоимость" ошибки на всём диапазоне напряжения) ---
    t_data = batch["t_n"]
    v_term, v1, v2, soc = model(t_data, batch["i_n"])
    w_v = batch.get("w_v")
    if w_v is not None:
        loss_data = torch.mean(w_v * (v_term - batch["v_n"]) ** 2)
    else:
        loss_data = torch.mean((v_term - batch["v_n"]) ** 2)
    loss_soc = torch.mean((soc - batch["soc_meas"]) ** 2)

    # --- ОДУ-резидуалы на коллокациях (D6: фиксированная Sobol-сетка) ---
    t_n = batch["t_col_n"].requires_grad_(True)
    _, v1c, v2c, socc = model(t_n, batch["i_col_n"])
    ones = torch.ones_like(v1c)
    dv1_dt_n = torch.autograd.grad(v1c, t_n, ones, create_graph=True)[0]
    dv2_dt_n = torch.autograd.grad(v2c, t_n, ones, create_graph=True)[0]
    dsoc_dt_n = torch.autograd.grad(socc, t_n, ones, create_graph=True)[0]

    dt = model.t_ref
    dv1 = dv1_dt_n * scale_v / dt
    dv2 = dv2_dt_n * scale_v / dt
    dsoc = dsoc_dt_n / dt

    i_phys = batch["i_phys_col"]
    r1 = dv1 + v1c * scale_v / (model.R1 * model.C1) - i_phys / model.C1
    r2 = dv2 + v2c * scale_v / (model.R2 * model.C2) - i_phys / model.C2
    q = max(batch.get("q_max_ah", 0.025), 1e-3)
    eta = lam.get("eta_c", 1.0)
    r_soc = dsoc + eta * i_phys / (3600.0 * q)

    loss_ode = torch.mean(r1 ** 2) + torch.mean(r2 ** 2) + torch.mean(r_soc ** 2)

    # начальные условия: V1(0)=V2(0)=0 на коллокациях
    j0 = (t_n == t_n.min()).nonzero(as_tuple=True)[0]
    if len(j0) > 0:
        loss_ic = torch.mean(v1c[j0] ** 2 + v2c[j0] ** 2)
    else:
        loss_ic = torch.tensor(0.0)

    total = (lam["lambda_data"] * loss_data + lam["lambda_ode"] * loss_ode
             + lam["lambda_soc"] * loss_soc + lam["lambda_ic"] * loss_ic)
    parts = {"data": float(loss_data), "ode": float(loss_ode),
             "soc": float(loss_soc), "ic": float(loss_ic)}
    return total, parts
