# -*- coding: utf-8 -*-
"""
End-to-end smoke test for the multilayer flow using config.json.
- Loads config.json
- Builds the multilayer S-matrix via Builder_config + StructureBuilder
- Prints basic metrics (freq range, |Gamma_in| stats)
- Optionally saves magnitude/phase plots to pictures/ (if matplotlib installed)

Run:
  python3 -m test.run_config_flow
"""
from __future__ import annotations
import os
from pathlib import Path
import datetime as _dt
import numpy as np

from Builder_config import load_json, build_from_config

# Optional plotting
try:
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


def _root_dir() -> Path:
    # This file is at <ROOT>/test/run_config_flow.py
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = _root_dir()
    cfg_path = root / 'config.json'
    if not cfg_path.exists():
        print(f"config.json not found at {cfg_path}")
        return 2

    cfg = load_json(str(cfg_path))
    out = build_from_config(cfg)

    omega: np.ndarray = out['omega']
    H_ref: np.ndarray = out['H_ref']  # Γ_in for half-space termination

    f = omega / (2 * np.pi)
    print("-- Build summary --")
    print(f"Freq points: {f.size}")
    step_kHz = ((f[1] - f[0]) / 1e3) if f.size > 1 else float('nan')
    print(f"Freq range: {f[0] / 1e6:.3f} → {f[-1] / 1e6:.3f} MHz  step: {step_kHz:.1f} kHz")
    mag = np.abs(H_ref)

    # 基线 |Γ|（首个界面，例如水-钢）
    baseline = compute_first_interface_baseline_gamma(cfg)
    if baseline is None:
        print("Baseline |Gamma|: <none>  (no 'interface' found in chain)")
        mag_delta = None
    else:
        print(f"Baseline |Gamma| (first interface): {baseline:.6f}")
        mag_delta = mag - baseline

    # 工程量：回波损耗与驻波比
    RL_dB = 20.0 * np.log10(np.clip(mag, 1e-12, None))
    SWR = (1.0 + mag) / np.clip(1.0 - mag, 1e-6, None)

    phase = np.unwrap(np.angle(H_ref))
    print(f"|Γ_in|: min={mag.min():.4f}, max={mag.max():.4f}")
    finite = np.isfinite(H_ref).all()
    print(f"All finite: {bool(finite)}")

    # ---------- 保存数据 (CSV + NPZ) ----------
    art_dir = root / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    cols = [f, H_ref.real, H_ref.imag, mag, phase, RL_dB, SWR]
    hdrs = ["f_Hz", "Re_Gamma", "Im_Gamma", "abs_Gamma", "phase_unwrap_rad", "RL_dB", "SWR"]

    if mag_delta is not None:
        cols.append(mag_delta)
        hdrs.append("abs_Gamma_delta_vs_first_interface")
    data = np.column_stack(cols)
    csv_fp = art_dir / f"run_config_{ts}.csv"
    npz_fp = art_dir / f"run_config_{ts}.npz"
    np.savetxt(csv_fp, data, delimiter=",", header=",".join(hdrs), comments="")
    np.savez(npz_fp, f=f, H_ref=H_ref, mag=mag, phase=phase, RL_dB=RL_dB, SWR=SWR,
             mag_delta=mag_delta, baseline=baseline)
    print(f"Saved: {csv_fp}")
    print(f"Saved: {npz_fp}")

    # Optionally plot
    if _HAVE_MPL:
        pic_dir = root / 'pictures'
        pic_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')

        def _plot_and_save(x_Hz, y, ylabel, title, suffix):
            x_MHz = x_Hz / 1e6
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(x_MHz, y)
            ax.set_xlim(x_MHz[0], x_MHz[-1])
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True)
            fig.tight_layout()
            out_fp = pic_dir / f"run_config_{suffix}_{ts}.png"
            fig.savefig(out_fp, dpi=150)
            plt.close(fig)
            print(f"Saved: {out_fp}")

        _plot_and_save(f, mag, '|Γ_in|', 'Input reflection magnitude', 'mag')
        _plot_and_save(f, mag ** 2, '|Γ_in|^2', 'Reflection power', 'power')
        _plot_and_save(f, phase, 'Phase (rad)', 'Input reflection phase (unwrapped)', 'phase')

        # 差分：|Γ_in| - |Γ_baseline|（首个界面）

        if mag_delta is not None:
            _plot_and_save(f, mag_delta,
                           r'$|\Gamma_{in}| - |\Gamma_{baseline}|$',
                           'Delta vs first-interface baseline',
                           'delta')

        # 回波损耗 RL(dB)
        _plot_and_save(f, RL_dB, 'RL (dB)', 'Return loss (RL)', 'RLdB')

    else:
        print("matplotlib not available; skipping plots.")

    return 0


def _Z_from_cfg(cfg, medium_name: str) -> float:
    """从 config 取介质特性，返回纵向声阻抗 Z = ρ c_p."""
    m = cfg["media"][medium_name]
    return float(m["rho"]) * float(m["c_p"])


def compute_first_interface_baseline_gamma(cfg) -> float | None:
    """
    从 chain 里找到第一个 interface，返回其 |Γ| 基线（频率无关），找不到则返回 None。
    """
    for item in cfg.get("chain", []):
        if item.get("kind") == "interface":
            left = item["left"];
            right = item["right"]
            ZL = _Z_from_cfg(cfg, left)
            ZR = _Z_from_cfg(cfg, right)
            Gamma = (ZR - ZL) / (ZR + ZL)
            return abs(Gamma)
    return None


if __name__ == '__main__':
    raise SystemExit(main())
