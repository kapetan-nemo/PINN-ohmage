"""dataset_engine.py — пайплайн реальных coin-cell данных.

Источники (реальные данные, синтетика отменена заказчиком):
  - Mendeley m8w8sjk3vm (LIR2025H, 45 ячеек, Landt CT3001A):
    zip 'Original Bat Data/batN/M.csv' — один CSV на цикл, столбцы Landt:
    'Cycle ID','Step ID','Step Name','Time(h:min:s.ms)','Voltage(V)','Current(mA)','Capacity(mAh)'
    Шаги: Rest / CCCV_Chg / CC_DChg.
  - Zenodo 15069341 (SINTEF CR2032, первичная Li-MnO2, разряд 11 мА) — parquet.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy.stats import qmc


@dataclass
class CellData:
    cell_id: str
    t: torch.Tensor        # (N,1) секунды от начала разрядного шага
    i: torch.Tensor        # (N,1) амперы, разряд > 0
    v: torch.Tensor        # (N,1) вольты (сглаженные)
    soc: torch.Tensor      # (N,1) из Capacity(mAh)/Q_max
    q_max_ah: float        # ёмкость этого цикла (разряд)
    cycle_n: int
    meta: dict


def savgol_clean(v: np.ndarray, window: int = 21, polyorder: int = 2) -> np.ndarray:
    window = int(window) | 1
    if len(v) < window or window <= polyorder + 2:
        return v
    return savgol_filter(v, window, polyorder)


def _parse_time_hms(s: pd.Series) -> np.ndarray:
    """'0:00:01.000' → секунды (float). Перезапускается в 0 каждый шаг/файл."""
    parts = s.astype(str).str.split(":", expand=True).astype(float)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def soc_from_capacity(cap_mah: np.ndarray, q_max_ah: float, soc_end: float = 0.0) -> np.ndarray:
    """SOC из накопленной разрядной ёмкости: SOC = 1 - cap/Q. Нормируем на факт цикла."""
    q = cap_mah[-1] / 1000.0
    if q <= 1e-6:
        return np.ones_like(cap_mah)
    return 1.0 - cap_mah / 1000.0 / q * (1.0 - soc_end)


def soc_coulomb_count(i_a: np.ndarray, t_s: np.ndarray, q_max_ah: float,
                      eta_c: float = 1.0, soc0: float = 1.0) -> np.ndarray:
    """SOC методом трапеций (для источников без счётчика ёмкости)."""
    q_cum = np.concatenate([[0.0], np.cumsum(0.5 * (i_a[1:] + i_a[:-1]) * np.diff(t_s))])
    return soc0 - eta_c * q_cum / (3600.0 * q_max_ah)


def sobol_collocation(n: int, t0: float, t1: float, seed: int = 42) -> np.ndarray:
    sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
    u = sampler.random(n).ravel()
    return t0 + u * (t1 - t0)


def segment_by_phase(df: pd.DataFrame, i_thr: float = 1e-4) -> pd.DataFrame:
    di = np.gradient(df["I_A"].values) / np.maximum(np.gradient(df["t_s"].values), 1e-9)
    phase = np.full(len(df), "rest", dtype=object)
    phase[df["I_A"].abs() < i_thr] = "rest"
    mask = df["I_A"].abs() >= i_thr
    phase[mask & (df["I_A"] > 0)] = "discharge"
    phase[mask & (df["I_A"] < 0)] = "charge"
    df = df.copy()
    df["phase"] = phase
    return df


class Normalizer:
    def __init__(self, t_ref: float = 3600.0, i_ref: float = 0.2,
                 v_min: float = 2.5, v_max: float = 4.4):
        self.t_ref, self.i_ref = t_ref, i_ref
        self.v_min, self.v_max = v_min, v_max

    @classmethod
    def fit(cls, cells: list[CellData]) -> "Normalizer":
        i_ref = max(max(float(c.i.abs().max()) for c in cells), 1e-3)
        v_all = torch.cat([c.v for c in cells])
        return cls(i_ref=i_ref, v_min=float(v_all.min()) - 0.05, v_max=float(v_all.max()) + 0.05)

    def t(self, t): return t / self.t_ref
    def i(self, i): return i / self.i_ref
    def v(self, v): return (v - self.v_min) / (self.v_max - self.v_min)
    def v_inv(self, v_n): return v_n * (self.v_max - self.v_min) + self.v_min


class CoinCellDataPipeline:
    """Загрузка LIR2025H (zip Landt) и SINTEF CR2032 (parquet)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    # ---------- LIR2025H ----------
    def load_lir2025h(self, zip_path: str, cells_sel: list[int] | None = None,
                      cycles: list[int] | None = None, max_points: int | None = None
                      ) -> list[CellData]:
        """cells_sel — номера batN; cycles — список номеров CSV (файлов-циклов).

        Возвращает список CellData: по одному на (ячейка, цикл).
        """
        zp = Path(zip_path)
        if cells_sel is None:
            with zipfile.ZipFile(zp) as zf:
                names = zf.namelist()
            bat_dirs = sorted({n.split("/")[2] for n in names
                               if n.startswith("Original Bat Data/bat") and "/" in n},
                              key=lambda s: int(s.replace("bat", "")))
            cells_sel = [int(d.replace("bat", "")) for d in bat_dirs]
        if cycles is None:
            cycles = [1, 25, 50, 75, 100]
        out = []
        with zipfile.ZipFile(zp) as zf:
            for b in cells_sel:
                for cy in cycles:
                    name = f"Original Bat Data/bat{b}/{cy}.csv"
                    if name not in zf.namelist():
                        continue
                    cell = self._landt_cycle_to_cell(zf.read(name), f"bat{b}_c{cy}",
                                                     cell_id=f"bat{b}", cycle_n=cy,
                                                     max_points=max_points)
                    if cell is not None:
                        out.append(cell)
        return out

    def _landt_cycle_to_cell(self, raw: bytes, key: str, cell_id: str, cycle_n: int,
                             max_points: int | None) -> CellData | None:
        df = pd.read_csv(io.BytesIO(raw))
        d = df[df["Step Name"] == "CC_DChg"]
        if len(d) < 50:
            return None
        t = _parse_time_hms(d["Time(h:min:s.ms)"]).values
        i = d["Current(mA)"].abs().values / 1000.0
        v = savgol_clean(d["Voltage(V)"].values.astype(float),
                         self.cfg["data"]["window"], self.cfg["data"]["polyorder"])
        cap = d["Capacity(mAh)"].values.astype(float)
        q_ah = cap[-1] / 1000.0
        if q_ah < 1e-4:
            return None
        soc = soc_from_capacity(cap, q_ah)
        if max_points and len(d) > max_points:
            idx = np.linspace(0, len(d) - 1, max_points).astype(int)
            t, i, v, soc = t[idx], i[idx], v[idx], soc[idx]
        return CellData(
            cell_id=cell_id, key=key, cycle_n=cycle_n, q_max_ah=float(q_ah),
            t=torch.tensor(t.reshape(-1, 1), dtype=torch.float32),
            i=torch.tensor(i.reshape(-1, 1), dtype=torch.float32),
            v=torch.tensor(v.reshape(-1, 1), dtype=torch.float32),
            soc=torch.tensor(np.clip(soc, 1e-3, 1.0).reshape(-1, 1), dtype=torch.float32),
            meta={"phase": "discharge", "n_raw": int(len(d))},
        ) if False else CellData(
            cell_id=cell_id, cycle_n=cycle_n, q_max_ah=float(q_ah),
            t=torch.tensor(t.reshape(-1, 1), dtype=torch.float32),
            i=torch.tensor(i.reshape(-1, 1), dtype=torch.float32),
            v=torch.tensor(v.reshape(-1, 1), dtype=torch.float32),
            soc=torch.tensor(np.clip(soc, 1e-3, 1.0).reshape(-1, 1), dtype=torch.float32),
            meta={"phase": "discharge", "n_raw": int(len(d)), "key": key},
        )

    @staticmethod
    def _discharge_run(d_cyc: pd.DataFrame, i_thr: float = 1e-5):
        """Длиннейший непрерывный участок разряда внутри цикла.

        Разряд определяется по тренду напряжения (V падает), а не по знаку тока:
        конвенция знака различается между форматами тестеров. Участок режется по
        паузам (> 60 с) и по сбросам счётчика фазовой ёмкости.
        """
        cur = d_cyc["current_ampere"].values
        act = np.abs(cur) > i_thr
        runs, start = [], None
        for j, a in enumerate(act):
            if a and start is None:
                start = j
            elif not a and start is not None:
                runs.append((start, j)); start = None
        if start is not None:
            runs.append((start, len(act)))
        best = None
        for a, b in runs:
            if b - a < 50:
                continue
            seg = d_cyc.iloc[a:b].reset_index(drop=True)
            t = seg["test_time_millisecond"].values / 1000.0
            if t[-1] - t[0] < 60:
                continue
            # сбросы счётчика ёмкости → отдельные под-участки
            cap = seg["phase_capacity_ampere_hour"].values
            dec = np.where(np.diff(cap) < -1e-6)[0]
            bounds = np.concatenate([[0], dec + 1, [len(seg)]])
            for k in range(len(bounds) - 1):
                sub = seg.iloc[bounds[k]:bounds[k + 1]].reset_index(drop=True)
                if len(sub) < 50:
                    continue
                tt = sub["test_time_millisecond"].values / 1000.0
                vv = sub["voltage_volt"].values.astype(float)
                if tt[-1] - tt[0] < 60:
                    continue
                slope = float(np.polyfit(tt, vv, 1)[0])
                if slope >= 0:      # не разряд
                    continue
                if best is None or len(sub) > len(best[1]):
                    best = (k, sub)
        return None if best is None else best[1]

    # ---------- IOC (CSV в .bdf: form + cycling) ----------
    def load_ioc(self, dir_path: str, cell_ids: list[str] | None = None,
                 max_points: int | None = None, i_thr: float = 1e-5
                 ) -> list[CellData]:
        """IOC-датасет: data/Dataset_IOC/IOC_XXX_YYZ/XXX_YYZ_cycling.bdf (CSV).

        Столбцы: test_time_millisecond, voltage_volt, current_ampere,
        cycle_dimensionless, phase_capacity_ampere_hour, ambient_temperature_celsius.
        Возвращает CellData на каждый (ячейка, цикл) — длиннейший разрядный сегмент.
        """
        root = Path(dir_path)
        folders = sorted(d for d in root.iterdir()
                         if d.is_dir() and d.name.startswith("IOC_"))
        if cell_ids:
            folders = [d for d in folders if d.name in cell_ids]
        out = []
        for folder in folders:
            csv = folder / f"{folder.name.replace('IOC_', '')}_cycling.bdf"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            df = df.sort_values("test_time_millisecond").reset_index(drop=True)
            for cy, d_cyc in df.groupby("cycle_dimensionless"):
                d = self._discharge_run(d_cyc, i_thr=i_thr)
                if d is None or len(d) < 50:
                    continue
                t = d["test_time_millisecond"].values / 1000.0
                i = d["current_ampere"].abs().values
                v_raw = d["voltage_volt"].values.astype(float)
                cap = d["phase_capacity_ampere_hour"].values.astype(float)
                if max_points and len(d) > max_points:
                    idx = np.linspace(0, len(d) - 1, max_points).astype(int)
                    t, i, v_raw, cap = t[idx], i[idx], v_raw[idx], cap[idx]
                v = savgol_clean(v_raw, self.cfg["data"]["window"], self.cfg["data"]["polyorder"])
                q_ah = max(float(cap[-1]), float(i.mean() * (t[-1] - t[0]) / 3600.0))
                if q_ah < 1e-4:
                    continue
                soc = 1.0 - cap / q_ah
                out.append(CellData(
                    cell_id=folder.name, cycle_n=int(cy), q_max_ah=float(q_ah),
                    t=torch.tensor(t.reshape(-1, 1), dtype=torch.float32),
                    i=torch.tensor(i.reshape(-1, 1), dtype=torch.float32),
                    v=torch.tensor(v.reshape(-1, 1), dtype=torch.float32),
                    soc=torch.tensor(np.clip(soc, 0.0, 1.0).reshape(-1, 1), dtype=torch.float32),
                    meta={"phase": "discharge", "T_mean": float(d["ambient_temperature_celsius"].mean()),
                          "key": f"{folder.name}_c{int(cy)}"},
                ))
        return out

    # ---------- SINTEF CR2032 ----------
    def load_cr2032_sintef(self, parquet_path: str, max_points: int | None = None) -> list[CellData]:
        df = pd.read_parquet(parquet_path)
        d = pd.DataFrame({
            "t_s": df["test_time_millisecond"].astype(float) / 1000.0,
            "I_A": -df["current_ampere"].astype(float),
            "V_V": df["voltage_volt"].astype(float),
        })
        d = d[d["I_A"] > 1e-5].reset_index(drop=True)
        if max_points and len(d) > max_points:
            idx = np.linspace(0, len(d) - 1, max_points).astype(int)
            d = d.iloc[idx].reset_index(drop=True)
        v = savgol_clean(d["V_V"].values, self.cfg["data"]["window"], self.cfg["data"]["polyorder"])
        q_max = max(d["I_A"].max() * (d["t_s"].iloc[-1] - d["t_s"].iloc[0]) / 3600.0 * 1.05, 1e-4)
        soc = soc_coulomb_count(d["I_A"].values, d["t_s"].values, q_max)
        return [CellData(
            cell_id="sintef_cr2032", cycle_n=1, q_max_ah=float(q_max),
            t=torch.tensor(d["t_s"].values.reshape(-1, 1), dtype=torch.float32),
            i=torch.tensor(d["I_A"].values.reshape(-1, 1), dtype=torch.float32),
            v=torch.tensor(v.reshape(-1, 1), dtype=torch.float32),
            soc=torch.tensor(np.clip(soc, 1e-3, 1.0).reshape(-1, 1), dtype=torch.float32),
            meta={"phase": "discharge", "n_points": len(d)},
        )]
