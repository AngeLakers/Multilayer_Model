# -*- coding: utf-8 -*-
"""
Minimal, reusable multilayer structure builder that *reuses* functions already
present in your Multilayer_Model repo. No re-definitions of your physics blocks.

It only wraps your existing S_interface / S_layer / S_impedance_sheet /
Redheffer star utilities into a small class for easy composition,
plus an auto adhesive helper (sheet vs. explicit by |k d|).

Drop this file next to your current modules and import in tests/scripts:

    from builder import StructureBuilder

Author: you
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Callable, Sequence
import numpy as np

# --- import *existing* blocks from your repo ---
# If your file/module name is different, adjust the import line accordingly.
from multilayer_smatrix import (
    S_interface,
    S_layer,
    S_impedance_sheet,
    star,
    fold_star,
    gamma_in_from_S,
)

SBlock = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

class StructureBuilder:
    """
    Small helper to assemble multilayer S-block chains using your existing functions.

    Usage
    -----
    sb = StructureBuilder(omega)
    sb.add_interface(ZL, ZR)
    sb.add_layer(c_p=1600.0, d=1e-3, causal=True, alpha0=..., n=1.2, omega0=...)
    sb.add_adhesive_auto(ZL, ZR, thickness=80e-6, rho=1200.0, E_storage=2e9, tan_delta=0.08,
                         c_p=2000.0, alpha0=..., n=1.2, f0_Hz=1e6, kd_thresh=0.1)
    S_tot = sb.build()
    Gamma_in = sb.gamma_in(S_tot)   # (Γ_L=0 half-space) => S11
    """

    def __init__(self, omega: np.ndarray):
        self.omega = np.asarray(omega, dtype=float)
        if self.omega.ndim != 1:
            raise ValueError("omega must be 1D vector")
        self._blocks: List[SBlock] = []

    # -------------------- basic blocks --------------------
    def add_interface(self, ZL: float, ZR: float):
        self._blocks.append(S_interface(ZL, ZR))
        return self

    def add_layer(self,
                  c_p: float,
                  d: float,
                  *,
                  causal: bool = False,
                  alpha0: Optional[float] = None,
                  n: Optional[float] = None,
                  omega0: Optional[float] = None,
                  alpha_np_per_m: Optional[np.ndarray] = None):
        """
        Add a uniform layer using your S_layer. Supports both (legacy) alpha_np_per_m path
        and (new) causal power-law parameters.
        """
        # Try new causal-signature first; fallback to legacy signature.
        try:
            if causal:
                if alpha0 is None or n is None or omega0 is None:
                    raise ValueError("causal=True requires alpha0, n, omega0")
                blk = S_layer(self.omega, c_p=c_p, d=d,
                              causal=True, alpha0=alpha0, n=n, omega0=omega0)
            else:
                blk = S_layer(self.omega, c_p=c_p, d=d, alpha_np_per_m=(0.0 if alpha_np_per_m is None else alpha_np_per_m))
        except TypeError:
            # Legacy-only implementation
            if causal:
                raise TypeError("Your S_layer does not support causal=... signature.")
            blk = S_layer(self.omega, c_p, d, (0.0 if alpha_np_per_m is None else alpha_np_per_m))
        self._blocks.append(blk)
        return self

    # -------------------- adhesive helpers --------------------
    def add_adhesive_sheet(self,
                           ZL: float,
                           ZR: float,
                           *,
                           m_prime: float,
                           K_n: Optional[float] = None,
                           tan_delta: Optional[float] = None,
                           R_prime: float = 0.0,
                           K_n_of_omega: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                           R_prime_of_omega: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        blk = S_impedance_sheet(self.omega, ZL, ZR,
                                m_prime=m_prime, R_prime=R_prime,
                                K_n=K_n, tan_delta=tan_delta,
                                K_n_of_omega=K_n_of_omega,
                                R_prime_of_omega=R_prime_of_omega)
        self._blocks.append(blk)
        return self

    def add_adhesive_explicit(self,
                              ZL: float,
                              ZR: float,
                              *,
                              thickness: float,
                              rho: float,
                              c_p: float,
                              alpha0: float,
                              n: float,
                              f0_Hz: float):
        Zadh = rho * c_p
        self._blocks.append(S_interface(ZL, Zadh))
        # explicit layer must be causal to remain consistent with attenuation model
        blk = S_layer(self.omega, c_p=c_p, d=thickness,
                      causal=True, alpha0=alpha0, n=n, omega0=2*np.pi*f0_Hz)
        self._blocks.append(blk)
        self._blocks.append(S_interface(Zadh, ZR))
        return self

    def add_adhesive_auto(self,
                          ZL: float,
                          ZR: float,
                          *,
                          thickness: float,
                          rho: float,
                          E_storage: float,
                          tan_delta: float,
                          R_prime: float = 0.0,
                          c_p: Optional[float] = None,
                          alpha0: Optional[float] = None,
                          n: Optional[float] = None,
                          f0_Hz: Optional[float] = None,
                          kd_thresh: float = 0.1):
        """
        Auto choose adhesive model:
          if c_p is None -> use sheet;
          else compute max(|k d|) = max(ω/c_p * thickness). If <= kd_thresh -> sheet; else explicit.
        Sheet parameters: m′=ρ·d, K_n=E′/d, tanδ, R′(optional).
        Explicit requires (c_p, alpha0, n, f0_Hz).
        """
        if c_p is None:
            m_prime = rho * thickness
            K_n = E_storage / thickness
            return self.add_adhesive_sheet(ZL, ZR, m_prime=m_prime, K_n=K_n,
                                           tan_delta=tan_delta, R_prime=R_prime)
        kd_max = float(np.max(np.abs(self.omega / float(c_p) * thickness)))
        if kd_max <= kd_thresh:
            m_prime = rho * thickness
            K_n = E_storage / thickness
            return self.add_adhesive_sheet(ZL, ZR, m_prime=m_prime, K_n=K_n,
                                           tan_delta=tan_delta, R_prime=R_prime)
        if (alpha0 is None) or (n is None) or (f0_Hz is None):
            raise ValueError("Explicit adhesive selected (|k d|>thresh) but alpha0/n/f0_Hz not provided.")
        return self.add_adhesive_explicit(ZL, ZR, thickness=thickness, rho=rho,
                                          c_p=c_p, alpha0=alpha0, n=n, f0_Hz=f0_Hz)

    # -------------------- build & utilities --------------------
    def build(self) -> SBlock:
        if not self._blocks:
            raise RuntimeError("No blocks added. Add interfaces/layers/adhesives before build().")
        return fold_star(self._blocks)

    def gamma_in(self, S_tot: SBlock, Gamma_L: Optional[np.ndarray] = None) -> np.ndarray:
        return gamma_in_from_S(S_tot, Gamma_L=Gamma_L)

    @staticmethod
    def energy_residual(S_tot: SBlock, ZL: float, ZR: float) -> float:
        S11, _, S21, _ = S_tot
        res = np.abs(np.abs(S11)**2 + (np.real(ZR)/np.real(ZL))*np.abs(S21)**2 - 1.0)
        return float(np.max(res))


# -------------------- convenience --------------------

def db_cm_MHzn_to_Np_m(alpha_db_per_cm_per_MHzn: float, n: float, f0_MHz: float) -> float:
    """Convert α in dB/(cm·MHz^n) to α0 [Np/m] at f0 (MHz)."""
    conv = (np.log(10.0) / 20.0) * 100.0
    return alpha_db_per_cm_per_MHzn * conv * (f0_MHz ** n)


# -------------------- example (remove in production) --------------------
if __name__ == "__main__":
    f = np.arange(0.1e6, 3.0e6+1e3, 1e3)
    w = 2*np.pi*f

    # impedances (examples)
    Z_water = 1000.0 * 1480.0
    Z_rubber = 1100.0 * 1600.0
    Z_metal  = 2780.0 * 6320.0

    sb = StructureBuilder(w)
    sb.add_interface(Z_water, Z_rubber)
    sb.add_layer(c_p=1600.0, d=1e-3, causal=False)
    sb.add_adhesive_auto(Z_rubber, Z_metal,
                         thickness=80e-6, rho=1200.0,
                         E_storage=2.0e9, tan_delta=0.08,
                         c_p=2000.0,
                         alpha0=db_cm_MHzn_to_Np_m(0.5, 1.2, 1.0), n=1.2, f0_Hz=1.0e6,
                         kd_thresh=0.1)
    sb.add_layer(c_p=6320.0, d=1e-3, causal=False)

    S_tot = sb.build()
    Gamma_in = sb.gamma_in(S_tot)
    print("Gamma_in shape:", Gamma_in.shape)
