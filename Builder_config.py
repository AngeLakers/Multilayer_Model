# -*- coding: utf-8 -*-
"""
Config → StructureBuilder glue layer.
- 仅调用你已有的 StructureBuilder（其内部再调用 multilayer_smatrix 的函数）。
- 不重复定义任何物理块。

用法：
    from builder_config import build_from_config, load_json
    cfg = load_json('config.json')
    out = build_from_config(cfg)
    omega, H_ref, S_tot = out['omega'], out['H_ref'], out['S_tot']

配置结构参见你先前的示例（media / layers / adhesives / chain）。
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import json
import numpy as np

# 只依赖你画布里的 StructureBuilder；其内部再调用 multilayer_smatrix 提供的函数
from builder import StructureBuilder


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _make_omega(freq_cfg: Dict[str, Any]) -> np.ndarray:
    if 'omega_rad_s' in freq_cfg:
        w = np.asarray(freq_cfg['omega_rad_s'], dtype=float)
        if w.ndim != 1:
            raise ValueError('frequency.omega_rad_s must be 1D')
        return w
    for k in ('start_Hz', 'stop_Hz', 'step_Hz'):
        if k not in freq_cfg:
            raise ValueError(f'frequency.{k} missing')
    f0, f1, df = map(float, (freq_cfg['start_Hz'], freq_cfg['stop_Hz'], freq_cfg['step_Hz']))
    if not (df > 0 and f1 > f0):
        raise ValueError('invalid frequency range')
    return 2 * np.pi * np.arange(f0, f1 + 1e-12, df)


def _Z(media: Dict[str, Any], name: str) -> float:
    m = media[name]
    return float(m['rho']) * float(m['c_p'])


def build_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    读取配置并用 StructureBuilder 组装，返回 {omega, H_ref, S_tot}。
    说明：
      - chain 严格按左→右顺序；
      - halfspace 作为终端（Γ_L=0），无需额外 block；
      - adhesive: mode∈{sheet, explicit, auto}；auto 需要 c_p 与 atten 参数。
    """
    omega = _make_omega(cfg['frequency'])
    media: Dict[str, Any] = cfg['media']
    layers_by_name = {L['name']: L for L in cfg.get('layers', [])}
    adhesives = cfg.get('adhesives', {})

    sb = StructureBuilder(omega)

    for item in cfg['chain']:
        kind = item['kind']
        if kind == 'interface':
            ZL = _Z(media, item['left'])
            ZR = _Z(media, item['right'])
            sb.add_interface(ZL, ZR)
        elif kind == 'layer':
            ref = item['ref']
            L = layers_by_name[ref]
            med = media[L['medium']]
            c_p = float(med['c_p']); d = float(L['thickness_m'])
            att = L.get('atten', None)
            if att is None:
                sb.add_layer(c_p=c_p, d=d, causal=False)
            else:
                sb.add_layer(c_p=c_p, d=d, causal=True,
                             alpha0=float(att['alpha0_Np_per_m']),
                             n=float(att['n']),
                             omega0=2*np.pi*float(att['f0_Hz']))
        elif kind == 'adhesive':
            ref = item['ref']
            A = adhesives[ref]
            ZL = _Z(media, item['left'])
            ZR = _Z(media, item['right'])
            mode = A['mode']
            if mode == 'sheet':
                rho = float(A['rho']); d = float(A['thickness_m'])
                E_st = float(A['E_storage_Pa']); td = float(A.get('tan_delta', 0.0))
                Rpr = float(A.get('Rprime_Pa_s_per_m', 0.0))
                sb.add_adhesive_sheet(ZL, ZR, m_prime=rho*d, K_n=E_st/d, tan_delta=td, R_prime=Rpr)
            elif mode == 'explicit':
                rho = float(A['rho']); d = float(A['thickness_m']); c_p = float(A['c_p'])
                att = A['atten']
                sb.add_adhesive_explicit(ZL, ZR, thickness=d, rho=rho, c_p=c_p,
                                          alpha0=float(att['alpha0_Np_per_m']),
                                          n=float(att['n']),
                                          f0_Hz=float(att['f0_Hz']))
            elif mode == 'auto':
                rho = float(A['rho']); d = float(A['thickness_m'])
                E_st = float(A['E_storage_Pa']); td = float(A.get('tan_delta', 0.0))
                Rpr = float(A.get('Rprime_Pa_s_per_m', 0.0))
                c_p = A.get('c_p', None)
                kd_thresh = float(item.get('kd_thresh', 0.1))
                if c_p is None:
                    sb.add_adhesive_sheet(ZL, ZR, m_prime=rho*d, K_n=E_st/d, tan_delta=td, R_prime=Rpr)
                else:
                    att = A.get('atten', None)
                    if att is None:
                        raise ValueError('adhesive.auto 需要 atten 用于显式层分支')
                    sb.add_adhesive_auto(ZL, ZR, thickness=d, rho=rho, E_storage=E_st, tan_delta=td, R_prime=Rpr,
                                         c_p=float(c_p),
                                         alpha0=float(att['alpha0_Np_per_m']),
                                         n=float(att['n']),
                                         f0_Hz=float(att['f0_Hz']),
                                         kd_thresh=kd_thresh)
            else:
                raise ValueError('adhesive.mode must be sheet|explicit|auto')
        elif kind == 'halfspace':
            # 终端为半空间：Γ_L=0；无需在链中加入 block。
            pass
        else:
            raise ValueError(f'unknown chain kind: {kind}')

    S_tot = sb.build()
    H_ref = sb.gamma_in(S_tot)  # 右端半空间 ⇒ Γ_L=0
    return {'omega': omega, 'H_ref': H_ref, 'S_tot': S_tot}
