# -*- coding: utf-8 -*-
"""
Run multilayer config, compute Γ_in plus energy decomposition and *baseline normalization*.

Outputs CSV/NPZ with columns:
  f_Hz, Re_Gamma, Im_Gamma, abs_Gamma, phase_unwrap_rad, RL_dB, SWR,
  R, T, A, eta_down_vs_baseline, eta_T_vs_baseline

Usage:
  python -m test.run_config_flow_baseline
"""
from __future__ import annotations
from pathlib import Path
import datetime as _dt
import numpy as np

# Ensure local imports resolve
import sys
HERE = Path(__file__).resolve().parent.parent if (Path(__file__).name == "run_config_flow_baseline.py") else Path.cwd()
sys.path.insert(0, str(HERE))

# Import your existing glue layer
from Builder_config import build_from_config, load_json  # type: ignore





def _root_dir():
    """Return the root directory of the project."""
    return Path(__file__).resolve().parent.parent

root = _root_dir()

def _Z_from_cfg(cfg: dict, medium_name: str) -> float:
    m = cfg["media"][medium_name]
    return float(m["rho"]) * float(m["c_p"])


def _find_ports(cfg: dict):
    """Return (left_medium_name, right_medium_name) for the entire chain."""
    left = None
    right = None
    for it in cfg.get("chain", []):
        if it.get("kind") == "interface" and left is None:
            left = it["left"]
        if it.get("kind") == "halfspace":
            right = it["medium"]
    if right is None:
        # Fallback to last interface's right if no halfspace
        for it in reversed(cfg.get("chain", [])):
            if it.get("kind") == "interface":
                right = it["right"]
                break
    return left, right


def _compute_tau_baseline(cfg: dict):
    """
    Choose the baseline *entry* interface for power transmission normalization.
    Priority:
      1) cfg["baseline_override"] = {"left": "...", "right": "..."}
      2) fallback to the *first* interface in chain
    Return (tau_base, left_name, right_name) or (None, None, None).
    """
    if "baseline_override" in cfg:
        left = cfg["baseline_override"]["left"]
        right = cfg["baseline_override"]["right"]
    else:
        left = right = None
        for it in cfg.get("chain", []):
            if it.get("kind") == "interface":
                left, right = it["left"], it["right"]
                break
        if left is None or right is None:
            return None, None, None
    ZL = _Z_from_cfg(cfg, left)
    ZR = _Z_from_cfg(cfg, right)
    tau = 4.0 * ZL * ZR / (ZL + ZR) ** 2
    return float(tau), left, right


def _ensure_dirs():
    out_dir = Path("artifacts")
    pic_dir = Path("pictures")
    out_dir.mkdir(exist_ok=True, parents=True)
    pic_dir.mkdir(exist_ok=True, parents=True)
    return out_dir, pic_dir


def main():
    root = _root_dir()
    cfg_path = root / 'config.json'
    if not cfg_path.exists():
        print(f"config.json not found at {cfg_path}")
        return 2

    cfg = load_json(str(cfg_path))
    out = build_from_config(cfg)

    out = build_from_config(cfg)
    omega = out["omega"]                   # (N,)
    H_ref = out["H_ref"]                   # (N,) complex
    S11, S12, S21, S22 = out["S_tot"]     # each (N,) complex

    f = omega / (2.0 * np.pi)
    mag = np.abs(H_ref)
    phase = np.unwrap(np.angle(H_ref))
    RL_dB = 20.0 * np.log10(np.clip(mag, 1e-12, None))
    SWR = (1.0 + mag) / np.clip(1.0 - mag, 1e-9, None)

    # Energy decomposition (pressure-normalized S; adjust by port impedances for power)
    left_med, right_med = _find_ports(cfg)
    ZL_port = _Z_from_cfg(cfg, left_med) if left_med else 1.0
    ZR_port = _Z_from_cfg(cfg, right_med) if right_med else 1.0

    R = np.abs(S11) ** 2
    T = np.abs(S21) ** 2 * (np.real(ZL_port) / np.real(ZR_port))
    A = 1.0 - R - T
    A = np.clip(A, 0.0, None)
    balance_err = np.max(np.abs(1.0 - (R + T + A)))

    # Baseline normalization
    tau_base, bL, bR = _compute_tau_baseline(cfg)
    if tau_base is None:
        eta_down = np.full_like(f, np.nan, dtype=float)
        eta_T = np.full_like(f, np.nan, dtype=float)
        baseline_info = "<none>"
    else:
        eta_down = (1.0 - R) / tau_base
        eta_T = T / tau_base
        baseline_info = f"{bL}->{bR}  tau_base={tau_base:.6f}"

    # I/O
    out_dir, pic_dir = _ensure_dirs()
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"run_config_with_baseline_{ts}"
    csv_fp = out_dir / f"{stem}.csv"
    npz_fp = out_dir / f"{stem}.npz"

    hdrs = [
        "f_Hz","Re_Gamma","Im_Gamma","abs_Gamma","phase_unwrap_rad","RL_dB","SWR",
        "R","T","A","eta_down_vs_baseline","eta_T_vs_baseline"
    ]
    data = np.column_stack([
        f, H_ref.real, H_ref.imag, mag, phase, RL_dB, SWR, R, T, A, eta_down, eta_T
    ])
    np.savetxt(csv_fp, data, delimiter=",", header=",".join(hdrs), comments="")

    np.savez(npz_fp,
             f=f, omega=omega, H_ref=H_ref, abs_Gamma=mag, phase_unwrap_rad=phase,
             RL_dB=RL_dB, SWR=SWR, R=R, T=T, A=A,
             eta_down=eta_down, eta_T=eta_T,
             meta=dict(baseline=baseline_info, balance_err=float(balance_err),
                       ZL_port=float(ZL_port), ZR_port=float(ZR_port)))

    print(f"[OK] Saved CSV → {csv_fp}")
    print(f"[OK] Saved NPZ → {npz_fp}")
    print(f"Baseline: {baseline_info}")
    print(f"Energy balance residual (max): {balance_err:.3e}")

    # Optional plots (no specific styles/colors; single-axes per figure)
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        def _plot(x, y, title, ylab, outname):
            plt.figure()
            plt.plot(x, y)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel(ylab)
            plt.title(title)
            figp = pic_dir / f"{outname}.png"
            plt.savefig(figp, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[FIG] {figp}")

        _plot(f, 1.0 - R, "Not-reflected power (1 - R)", "1 - R", f"{stem}_one_minus_R")
        _plot(f, eta_down, "η_down = (1 - R) / τ_base", "eta_down_vs_baseline", f"{stem}_eta_down")
        _plot(f, eta_T, "η_T = T / τ_base", "eta_T_vs_baseline", f"{stem}_eta_T")
    except Exception as e:
        print(f"[WARN] Plotting skipped: {e}")

if __name__ == "__main__":
    raise SystemExit(main())
