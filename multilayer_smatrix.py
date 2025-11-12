# -*- coding: utf-8 -*-
# multilayer_smatrix.py
# 1D 正向入射、压力幅值规范。支持：幂律衰减、显式层、界面、阻抗片、Redheffer星积。
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Callable


# ---------- 衰减模型 ----------
def alpha_powerlaw(omega: np.ndarray, alpha0: float, n: float, omega0: float) -> np.ndarray:
    """
    幂律衰减： alpha(ω) = alpha0 * (ω/ω0)^n    [单位：Np/m]
    推荐 omega0 取工作带宽的中心角频率，利于数值稳定。
    """
    x = np.maximum(omega, 1e-16)  # 避免0频率
    return alpha0 * (x / omega0) ** n

def gamma_from_params(omega: np.ndarray, c_p: float, d: float,
                      alpha_np_per_m: np.ndarray) -> np.ndarray:
    """
    复传播常数 γ = α(ω) + j*ω/c_p
    """
    return alpha_np_per_m + 1j * omega / c_p

# ---------- 基础构件：S-块 ----------
def S_interface(Za: float, Zb: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    界面块（压力幅度规范）
    返回四个标量/数组元素：S11, S12, S21, S22
    """
    r = (Zb - Za) / (Zb + Za)
    tab = 2 * Zb / (Zb + Za)
    tba = 2 * Za / (Zb + Za)
    # 让它们都成为0维 array，便于与频率维广播
    return np.asarray(r), np.asarray(tba), np.asarray(tab), np.asarray(-r)


def complex_k_powerlaw(omega: np.ndarray,
                       c0: float,
                       alpha0: float,
                       n: float,
                       omega0: float,
                       omega_ref: Optional[float] = None) -> np.ndarray:
    """
    因果幂律介质的复波数 k(ω) = ω/c0 + Δβ(ω) - i α(ω)
    α(ω) = α0 * (ω/ω0)^n  （单位：Np/m）
    Δβ(ω) = α(ω) * cot(π n / 2)        (n ∈ (0,2), n != 1)
    n = 1 时：Δβ(ω) = (2 α0 / π) * ω * ln(ω / ω_ref)
    """
    # 幂律衰减（Np/m）
    # --- 合法性：n 必须在 (0,2)，否则因果模型无定义/数值病态 ---

    if not (0.0 < n < 2.0):

        raise ValueError(f"causal power-law requires 0<n<2; got n={n}")
    alpha = alpha0 * (omega / omega0) ** n

    # 因果色散修正
    if abs(n - 1.0) > 1e-8:
        # Δβ = α * cot(π n / 2)；cot(x) = 1/tan(x)
        denom = np.tan(np.pi * n / 2.0)
        if abs(denom) < 1e-8:
             # 极近奇点，改抛错而不是装死
            raise ValueError(f"tan(pi*n/2) ~ 0 at n={n}, unstable for causal model")
        beta_corr = alpha / denom
    else:
        # n=1 特例：对数色散
        if omega_ref is None:
            # 取频带中心作为参考，保证相位连续
            omega_ref = float(np.median(omega))
        beta_corr = (2.0 * alpha0 / np.pi) * omega * np.log(omega / omega_ref)

    # 复波数：相位项 + 色散修正 - i*衰减
    k = (omega / c0) + beta_corr - 1j * alpha
    return k


def S_layer(omega: np.ndarray,
            c_p: float,
            d: float,
            alpha_np_per_m: Optional[np.ndarray] = None,
            *,
            causal: bool = False,
            alpha0: Optional[float] = None,
            n: Optional[float] = None,
            omega0: Optional[float] = None,
            omega_ref: Optional[float] = None
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    均匀各向同性纵波层的 S-矩阵：
        S = [[0, E],
             [E, 0]],   其中  E(ω) = exp(-i k(ω) d)
    - 非因果(默认)：使用 alpha_np_per_m 仅作幅度衰减，k = ω/c_p - i*alpha
    - 因果(causal=True)：启用因果幂律，k = ω/c_p + Δβ(ω) - i*α(ω)

    参数
    ----
    omega : [F] 角频率 (rad/s)
    c_p   : 相速度 (m/s)
    d     : 厚度 (m)
    alpha_np_per_m : [F] 或 None，若给定，在 causal=False 时用于衰减（Np/m）
    causal: 是否启用因果幂律模型
    alpha0, n, omega0: 因果幂律所需参数（Np/m @ ω0，幂指数 n，参考角频率 ω0）
    omega_ref: n=1 情况下对数色散的参考角频率（缺省取频带中位数）

    返回
    ----
    (S11, S12, S21, S22) 其中 S11=S22=0, S12=S21=E
    """
    omega = np.asarray(omega, dtype=float)

    if causal:
        if (alpha0 is None) or (n is None) or (omega0 is None):
            raise ValueError("causal=True 需提供 alpha0, n, omega0（单位：alpha0 为 Np/m 对应 ω0）。")
        # 使用因果幂律生成复波数
        k = complex_k_powerlaw(omega, c0=c_p, alpha0=alpha0, n=n, omega0=omega0, omega_ref=omega_ref)
        E = np.exp(-1j * k * d)  # 注意此处 k 已含 -i*α 项
    else:
        # 兼容原逻辑：仅幅度衰减（若未提供则视作零衰减）
        if alpha_np_per_m is None:
            alpha = 0.0
        else:
            alpha = np.asarray(alpha_np_per_m, dtype=float)
        # gamma = alpha + i*k ；E = exp(-gamma d) = exp(-α d) * exp(-i k d)
        k_real = omega / c_p
        E = np.exp(-(alpha + 1j * k_real) * d)

    # 组装 S（各频点独立，直接返回 4 个 [F] 向量）
    Z = np.zeros_like(E, dtype=complex)
    return Z, E, E, Z


# ---------- 阻抗片（等效二端口，以 T 域注入后转 S） ----------
def S_impedance_sheet(omega: np.ndarray, ZL: float, ZR: float,
                      m_prime: float = 0.0, R_prime: float = 0.0,
                      K_n: Optional[float] = None,
                      tan_delta: Optional[float] = None,
                      K_n_of_omega: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                      R_prime_of_omega: Optional[Callable[[np.ndarray], np.ndarray]] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-port S of a *series* impedance sheet placed between media with impedances (ZL, ZR).
    Z_sheet(ω) = jω m′ + R′(ω) + K_n*(ω)/(jω), where K_n*(ω) = K_n·(1 + j tanδ) if scalar.

    Parameters
    ----------
    omega : array-like [F]
    ZL, ZR : float
        Reference impedances of the adjacent media.
    m_prime : float
        Areal mass density ρ_a d_a [kg/m²].
    R_prime : float
        Additional real loss per area [Pa·s/m] (constant part). Use R_prime_of_omega for frequency dependence.
    K_n : float, optional
        Normal stiffness per area [Pa/m] for the elastic part.
    tan_delta : float, optional
        Loss factor of the stiffness (Kelvin–Voigt style). If provided with K_n, K_n*(ω)=K_n (1 + j tanδ).
    K_n_of_omega : callable, optional
        If provided, overrides (K_n, tan_delta) with an explicit frequency-dependent complex stiffness per area.
    R_prime_of_omega : callable, optional
        If provided, adds a frequency-dependent real loss term to R′.

    Returns
    -------
    (S11, S12, S21, S22) : 4 arrays of shape [F]
    """
    omega = np.asarray(omega, dtype=float)

    # Build complex stiffness per area K*(ω)
    if K_n_of_omega is not None:
        Kstar = np.asarray(K_n_of_omega(omega), dtype=complex)
    else:
        if K_n is None:
            Kstar = 0.0
        else:
            td = 0.0 if tan_delta is None else tan_delta
            Kstar = K_n * (1.0 + 1j * td)

    # Real loss term R′(ω)
    if R_prime_of_omega is not None:
        Rw = R_prime + np.asarray(R_prime_of_omega(omega), dtype=float)
    else:
        Rw = float(R_prime)

    # Series impedance of the sheet across frequency
    Kstar = np.asarray(Kstar, dtype=complex)
    Z_sheet = 1j * omega * m_prime + Rw + (Kstar / (1j * omega) if np.any(Kstar) else 0.0)

    # Convert to T, then to S referenced to (ZL, ZR)
    F = omega.shape[0]
    T = np.zeros((F, 2, 2), dtype=complex)
    T[:, 0, 0] = 1.0
    T[:, 1, 1] = 1.0
    T[:, 0, 1] = Z_sheet
    T[:, 1, 0] = 0.0
    return T_to_S(T, ZL, ZR)

# ---------- T ↔ S 转换（用于阻抗片等“二端口”从 T 域注入） ----------

def T_to_S(T: np.ndarray, ZL: float, ZR: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将 T 矩阵（形状 [F,2,2] 或 [2,2]）转为 S 矩阵（四个 [F] 或标量）。
    端口参考阻抗分别为 ZL (左), ZR (右)。
    公式来自微波网络的标准转换（压力-速度的两端口同理）。
    """
    # 广播成 [F,2,2]
    T = np.asarray(T)
    if T.ndim == 2:
        T = np.broadcast_to(T, (1,) + T.shape)
    A, B, C, D = T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1]
    ZL = np.asarray(ZL); ZR = np.asarray(ZR)
    if ZL.ndim == 0: ZL = np.broadcast_to(ZL, A.shape)
    if ZR.ndim == 0: ZR = np.broadcast_to(ZR, A.shape)
    denom = (A + B / ZR + C * ZL + D)
    S11 = (A + B / ZR - C * ZL - D) / denom
    S21 = 2.0 / denom
    S12 = 2.0 * (A * D - B * C) / denom
    S22 = (-A + B / ZR - C * ZL + D) / denom
    return S11, S12, S21, S22

def S_to_T(S: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ZL: float, ZR: float) -> np.ndarray:
    """
    需要时可用；这里保留以备扩展。
    """
    S11, S12, S21, S22 = S
    denom = S21
    A = ((1 + S11) * (1 - S22) + S12 * S21) / (2 * S21)
    B = ZR * ((1 + S11) * (1 + S22) - S12 * S21) / (2 * S21)
    C = (1 / ZL) * ((1 - S11) * (1 - S22) - S12 * S21) / (2 * S21)
    D = ((1 - S11) * (1 + S22) + S12 * S21) / (2 * S21)
    T = np.stack([np.stack([A, B], -1), np.stack([C, D], -1)], -2)  # [F,2,2]
    return T
# ---------- Redheffer 星积 ----------

def star(SA: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
         SB: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
         eps: float = 1e-18) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    SA ⋆ SB ，两块连接：SA 在左、SB 在右。
    都是 (S11,S12,S21,S22)，每个是 [F] 复数组或标量（将广播）。
    """
    A11, A12, A21, A22 = SA
    B11, B12, B21, B22 = SB
    # 广播
    A11, A12, A21, A22 = np.broadcast_arrays(A11, A12, A21, A22)
    B11, B12, B21, B22 = np.broadcast_arrays(B11, B12, B21, B22)
    I = 1.0
    D1 = 1.0 - A22 * B11
    D2 = 1.0 - B11 * A22
    # 正则化以避免频带近极点的数值问题
    D1 = D1 + eps * 1j
    D2 = D2 + eps * 1j

    S11 = A11 + A12 * B11 / D1 * A21
    S12 = A12 / D2 * B12
    S21 = B21 / D1 * A21
    S22 = B22 + B21 * A22 / D2 * B12
    return (S11, S12, S21, S22)

def fold_star(blocks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
              eps: float = 1e-18):
    """
    从左到右折叠一串 S-块
    """
    S = blocks[0]
    for b in blocks[1:]:
        S = star(S, b, eps=eps)
    return S

# ---------- 输出核 ----------
def gamma_in_from_S(S_tot: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                    Gamma_L: Optional[np.ndarray] = None, eps: float = 1e-18) -> np.ndarray:
    """
    入端等效反射 Γ_in = S11 + S12 Γ_L (1 - S22 Γ_L)^-1 S21
    若右端多层已并入，取 Γ_L = 0 ⇒ Γ_in = S11。
    """
    S11, S12, S21, S22 = S_tot
    if Gamma_L is None:
        return S11
    denom = 1.0 - S22 * Gamma_L + 1j * eps
    return S11 + S12 * Gamma_L / denom * S21

def Teff_from_S(S_tot: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                Gamma_L: Optional[np.ndarray] = None, eps: float = 1e-18) -> np.ndarray:
    """
    透射有效核：T_eff = (1 - S22 Γ_L)^-1 S21；半无限匹配右端时 Γ_L = 0 ⇒ T_eff = S21。
    """
    S11, S12, S21, S22 = S_tot
    if Gamma_L is None:
        return S21
    denom = 1.0 - S22 * Gamma_L + 1j * eps
    return S21 / denom

# 新增：压力归一 S → 功率系数 R/T/A
def compute_RTA(S_tot, ZL, ZR):
    S11, S12, S21, S22 = S_tot
    R = np.abs(S11)**2
    # 端口功率修正：I = |p|^2/(2 Re{Z})
    T = np.abs(S21)**2 * (np.real(ZL) / np.real(ZR))
    A = 1.0 - R - T
    return R, T, np.maximum(A, 0.0)


# ---------- 能量核对（无耗模式） ----------
def check_energy_conservation(S_tot, ZL, ZR):
    S11, S12, S21, S22 = S_tot
    return np.max(np.abs(np.abs(S11)**2 + (np.real(ZR)/np.real(ZL))*np.abs(S21)**2 - 1.0))

# ----------  极小值自动标注：反射自动找谷值，给出频率与半高宽----------

def find_minima(f, mag):
    # 简单相邻比较；可换成 scipy.signal.find_peaks(-mag)
    idx = (mag[1:-1] < mag[:-2]) & (mag[1:-1] < mag[2:])
    idx = np.where(idx)[0] + 1
    return f[idx], mag[idx]



