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

    f = omega / (2*np.pi)
    print("-- Build summary --")
    print(f"Freq points: {f.size}")
    print(f"Freq range: {f[0]/1e6:.3f} → {f[-1]/1e6:.3f} MHz  step: {(f[1]-f[0])/1e3:.1f} kHz")
    mag = np.abs(H_ref)
    phase = np.unwrap(np.angle(H_ref))
    print(f"|Γ_in|: min={mag.min():.4f}, max={mag.max():.4f}")
    finite = np.isfinite(H_ref).all()
    print(f"All finite: {bool(finite)}")

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
        _plot_and_save(f, mag**2, '|Γ_in|^2', 'Reflection power', 'power')
        _plot_and_save(f, phase, 'Phase (rad)', 'Input reflection phase (unwrapped)', 'phase')
    else:
        print("matplotlib not available; skipping plots.")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

