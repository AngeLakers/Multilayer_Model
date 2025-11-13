"""
Air–steel–air 1D single-layer plate under normal incidence, lossless.
Approximate free–free boundary ⇒ thickness resonances at f_n = n * c_L / (2 d), n=1,2,3.
This test validates the S-matrix kernel on a simple uniform plate by comparing
numerically found transmission peaks with the analytic resonance frequencies.

This version follows the repo's config-driven flow (like run_config_flow):
- Build an in-memory config dict (media/layers/chain, with omega_rad_s)
- Call Builder_config.build_from_config(cfg)
- Use S_tot[2] as transmission amplitude (right half-space ⇒ T_eff = S21)
"""
#### 「无耗散 air–steel–air 一维单层厚度共振问题上，你的实现质量已经接近“机器误差主导”级别。」

from __future__ import annotations
import numpy as np

# Make sure the project root is on sys.path when running this file directly
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Use the repo's config glue
from Builder_config import build_from_config


def _find_peaks_simple(y: np.ndarray) -> np.ndarray:
    """Return indices of strict local maxima (no scipy)."""
    if y.size < 3:
        return np.array([], dtype=int)
    y0 = y[:-2]
    y1 = y[1:-1]
    y2 = y[2:]
    mask = (y1 > y0) & (y1 > y2)
    idx = np.nonzero(mask)[0] + 1
    return idx.astype(int)


def _make_cfg_for_air_steel_air(omega: np.ndarray, d: float,
                                rho_air: float, c_air: float,
                                rho_steel: float, cL_steel: float) -> dict:
    return {
        "frequency": {"omega_rad_s": omega},
        "media": {
            "air":   {"rho": float(rho_air),  "c_p": float(c_air)},
            "steel": {"rho": float(rho_steel), "c_p": float(cL_steel)}
        },
        "layers": [
            {"name": "steel_plate", "medium": "steel", "thickness_m": float(d)}
        ],
        "adhesives": {},
        "chain": [
            {"kind": "interface", "left": "air", "right": "steel"},
            {"kind": "layer", "ref": "steel_plate"},
            {"kind": "interface", "left": "steel", "right": "air"},
            {"kind": "halfspace", "medium": "air"}
        ]
    }


def test_uniform_plate_thickness_resonance_match_analytic():
    # Materials (SI units)
    rho_air = 1.2
    c_air = 340.0
    rho_steel = 7850.0
    cL_steel = 5900.0
    d = 2.0e-3  # 2 mm

    # Frequency axis: cover up to 8th mode with a bit of headroom
    n_high = 8
    f1 = cL_steel / (2.0 * d)
    f_start = 0.1e6
    f_stop = max(5.0e6, 1.02 * n_high * f1)  # ensure >= 8th mode
    n_point = 16384  # at least
    f = np.linspace(f_start, f_stop, n_point)
    omega = 2.0 * np.pi * f

    # Build config and run the flow
    cfg = _make_cfg_for_air_steel_air(omega, d, rho_air, c_air, rho_steel, cL_steel)
    out = build_from_config(cfg)

    # Transmission amplitude (right half-space termination ⇒ T_eff = S21)
    S11, S12, S21, S22 = out["S_tot"]
    T_mag = np.abs(S21)

    # Simple peak finding on |T|
    peak_idx = _find_peaks_simple(T_mag)
    assert peak_idx.size > 0, "No peaks found in transmission magnitude."
    f_peaks = f[peak_idx]

    # Analytic thickness resonances: f_n = n c_L / (2 d) for n=1..8
    n_modes = np.arange(1, n_high + 1, dtype=int)
    f_theory = n_modes * cL_steel / (2.0 * d)

    # For each analytic f_n, find the nearest numerical peak
    f_numeric = np.empty_like(f_theory)
    for i, ft in enumerate(f_theory):
        j = int(np.argmin(np.abs(f_peaks - ft)))
        f_numeric[i] = f_peaks[j]

    rel_err = np.abs(f_numeric - f_theory) / f_theory

    ok = np.all(rel_err < 0.03)
    if not ok:
        # Print helpful diagnostics on failure
        print("f_theory_Hz:", f_theory)
        print("f_numeric_Hz:", f_numeric)
        print("rel_err:", rel_err)
    assert ok, "Uniform plate resonance frequency mismatch exceeds 3% for one or more modes."


if __name__ == "__main__":
    # Optional: run this file directly to see printed diagnostics even when the test would pass.
    rho_air = 1.2
    c_air = 340.0
    rho_steel = 7850.0
    cL_steel = 5900.0
    d = 2.0e-3

    n_high = 8
    f1 = cL_steel / (2.0 * d)
    f = np.linspace(0.1e6, max(5.0e6, 1.02 * n_high * f1), 16384)
    omega = 2.0 * np.pi * f

    cfg = _make_cfg_for_air_steel_air(omega, d, rho_air, c_air, rho_steel, cL_steel)
    out = build_from_config(cfg)
    S21 = out["S_tot"][2]

    T_mag = np.abs(S21)
    peak_idx = _find_peaks_simple(T_mag)
    f_peaks = f[peak_idx]

    n_modes = np.arange(1, n_high + 1, dtype=int)
    f_theory = n_modes * cL_steel / (2.0 * d)
    f_numeric = np.array([f_peaks[np.argmin(np.abs(f_peaks - ft))] for ft in f_theory])
    rel_err = np.abs(f_numeric - f_theory) / f_theory

    print("Found peaks (first 10):", f_peaks[:10], "...")
    print("f_theory_Hz:", f_theory)
    print("f_numeric_Hz:", f_numeric)
    print("rel_err:", rel_err)
