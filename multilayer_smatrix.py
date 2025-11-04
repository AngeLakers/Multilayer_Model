# -*- coding: utf-8 -*-
# multilayer_smatrix.py
# 1D 正向入射、压力幅值规范。支持：幂律衰减、显式层、界面、阻抗片、Redheffer星积。

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

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
    return (np.asarray(r), np.asarray(tba), np.asarray(tab), np.asarray(-r))

def S_layer(omega: np.ndarray, c_p: float, d: float,
            alpha_np_per_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    均匀层块： S = [[0, E],[E, 0]] ，E = exp(-γ d)
    """
    gamma = gamma_from_params(omega, c_p, d, alpha_np_per_m)
    E = np.exp(-gamma * d)   # 形状: [F]
    Z = np.zeros_like(E, dtype=complex)
    return (Z, E, E, Z)

# ---------- 阻抗片（等效二端口，以 T 域注入后转 S） ----------

def S_impedance_sheet(omega: np.ndarray, ZL: float, ZR: float,
                      m_prime: float = 0.0, R_prime: float = 0.0,
                      K_n: Optional[float] = None, tan_delta: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    串联界面阻抗片：
      Z_sheet(ω) = jω m' + R' + K_n*(ω)/(jω), 其中 K_n*(ω) = K_n * (1 + j*tanδ)（若给定）
    该元件的 T 矩阵： [[1, Z_sheet],[0,1]] ，端口参考为左右介质阻抗 ZL, ZR
    返回相应的 S-矩阵。
    """
    Kstar = 0.0
    if K_n is not None:
        if tan_delta is None: tan_delta = 0.0
        Kstar = K_n * (1.0 + 1j * tan_delta)
    Z_sheet = 1j * omega * m_prime + R_prime + (Kstar / (1j * omega) if K_n is not None else 0.0)
    # 组装 T，再转 S
    T = np.zeros((omega.shape[0], 2, 2), dtype=complex)
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

# ---------- 示例：搭建你的结构（钢-环氧-橡胶-环氧-阻抗片-环氧-橡胶-环氧-钢） ----------
@dataclass
class LayerParam:
    rho: float
    c_p: float
    d: float
    alpha0: float = 0.0   # 幂律 A
    n: float = 1.0        # 幂律 n
    # 若需要更复杂的因果模型，可自行扩展为 c(ω)

def build_structure_S(omega: np.ndarray,
                      left_medium: LayerParam,   # 左半空间（例如水），仅需 rho,c_p
                      right_medium: LayerParam,  # 右半空间或把右侧并入后这里不用
                      steel1: LayerParam, epoxy1: LayerParam, rubber1: LayerParam, epoxy2: LayerParam,
                      sheet_params: Dict[str, float],  # {'m_prime':...,'R_prime':...,'K_n':...,'tan_delta':...} 可只给前两项
                      epoxy3: LayerParam, rubber2: LayerParam, epoxy4: LayerParam, steel2: LayerParam,
                      omega0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回整条链的 S_tot（四个 [F] 复数组）
    """
    # 介质阻抗
    Z = lambda p: p.rho * p.c_p
    # 各层衰减（Np/m）
    a = lambda p: alpha_powerlaw(omega, p.alpha0, p.n, omega0)

    blocks = []

    # 左端界面 L|S1
    blocks.append(S_interface(Z(left_medium), Z(steel1)))
    # S1 层
    blocks.append(S_layer(omega, steel1.c_p, steel1.d, a(steel1)))
    # S1|EP1
    blocks.append(S_interface(Z(steel1), Z(epoxy1)))
    # EP1
    blocks.append(S_layer(omega, epoxy1.c_p, epoxy1.d, a(epoxy1)))
    # EP1|R1
    blocks.append(S_interface(Z(epoxy1), Z(rubber1)))
    # R1
    blocks.append(S_layer(omega, rubber1.c_p, rubber1.d, a(rubber1)))
    # R1|EP2
    blocks.append(S_interface(Z(rubber1), Z(epoxy2)))
    # EP2
    blocks.append(S_layer(omega, epoxy2.c_p, epoxy2.d, a(epoxy2)))

    # EP2 | SHEET | EP3  —— 把阻抗片放在 EP2 与 EP3 之间
    # 先界面 EP2|EP3 的“中间”替换成阻抗片：用 T_sheet 转 S，再星积
    S_sheet = S_impedance_sheet(
        omega, ZL=Z(epoxy2), ZR=Z(epoxy2),  # 阻抗片两侧端口参考：可取与相邻介质一致；若两侧不同介质也可用 ZL,ZR 不同
        m_prime=sheet_params.get('m_prime', 0.0),
        R_prime=sheet_params.get('R_prime', 0.0),
        K_n=sheet_params.get('K_n', None),
        tan_delta=sheet_params.get('tan_delta', None)
    )
    # 把 S_sheet 直接接在 EP2 末端，然后再接 EP3
    blocks.append(S_sheet)

    # EP2|EP3 物理上依然是“同材界面”，若希望仅保留阻抗片的作用，下面这一步界面可省略；
    # 若阻丝与胶材存在真实过渡/界面，你也可以保留一个 EP2->EP3 的界面块：
    # blocks.append(S_interface(Z(epoxy2), Z(epoxy3)))   # 视物理需要选择
    # EP3
    blocks.append(S_layer(omega, epoxy3.c_p, epoxy3.d, a(epoxy3)))

    # EP3|R2
    blocks.append(S_interface(Z(epoxy3), Z(rubber2)))
    # R2
    blocks.append(S_layer(omega, rubber2.c_p, rubber2.d, a(rubber2)))
    # R2|EP4
    blocks.append(S_interface(Z(rubber2), Z(epoxy4)))
    # EP4
    blocks.append(S_layer(omega, epoxy4.c_p, epoxy4.d, a(epoxy4)))
    # EP4|S2
    blocks.append(S_interface(Z(epoxy4), Z(steel2)))
    # S2
    blocks.append(S_layer(omega, steel2.c_p, steel2.d, a(steel2)))
    # S2|R
    blocks.append(S_interface(Z(steel2), Z(right_medium)))

    # 折叠
    S_tot = fold_star(blocks)
    return S_tot

# ---------- 使用示意（把数值替换掉即可） ----------
if __name__ == "__main__":
    # 频率轴
    fmin, fmax, N = 0.5e6, 5e6, 4096
    f = np.linspace(fmin, fmax, N)
    omega = 2*np.pi*f
    omega0 = 2*np.pi*((fmin+fmax)/2)

    # 示例层参数（请替换为你的数值；钢的衰减可以设 alpha0=0）
    water = LayerParam(rho=1000, c_p=1480, d=0.0)  # 左端若为半空间，只需 rho,c_p
    steel1 = LayerParam(rho=7850, c_p=5900, d=0.002, alpha0=0.0, n=1.0)
    epoxy1 = LayerParam(rho=1200, c_p=2500, d=1e-4, alpha0=5.0, n=1.2)  # 示例
    rubber1 = LayerParam(rho=1100, c_p=1600, d=6e-4, alpha0=10.0, n=1.3)
    epoxy2 = LayerParam(rho=1200, c_p=2500, d=1e-4, alpha0=5.0, n=1.2)

    sheet = dict(m_prime=0.2, R_prime=5.0)  # 面密度 kg/m^2、面阻 Pa·s/m；先从两参开始

    epoxy3 = LayerParam(rho=1200, c_p=2500, d=1e-4, alpha0=5.0, n=1.2)
    rubber2 = LayerParam(rho=1100, c_p=1600, d=6e-4, alpha0=10.0, n=1.3)
    epoxy4 = LayerParam(rho=1200, c_p=2500, d=1e-4, alpha0=5.0, n=1.2)
    steel2 = LayerParam(rho=7850, c_p=5900, d=0.002, alpha0=0.0, n=1.0)
    right = LayerParam(rho=1000, c_p=1480, d=0.0)  # 右端若为水半无限

    S_tot = build_structure_S(
        omega, left_medium=water, right_medium=right,
        steel1=steel1, epoxy1=epoxy1, rubber1=rubber1, epoxy2=epoxy2,
        sheet_params=sheet, epoxy3=epoxy3, rubber2=rubber2, epoxy4=epoxy4, steel2=steel2,
        omega0=omega0
    )

    # 回波与透射核（右端并入半无限 ⇒ Γ_L=0）
    Gamma_in = gamma_in_from_S(S_tot, Gamma_L=None)
    T_eff = Teff_from_S(S_tot, Gamma_L=None)

    # 简单检查：被动性（|Γ|≤1）与形状
    assert np.all(np.abs(Gamma_in) <= 1.0000001 + 1e-6), "Passive check failed (tune eps or params)."
    print("Gamma_in shape:", Gamma_in.shape, "T_eff shape:", T_eff.shape)
