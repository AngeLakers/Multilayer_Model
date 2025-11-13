# Multilayer_Model

一维多层声学/超声结构的 S 矩阵建模与快速拼接：界面、均匀层（支持因果幂律衰减与 n=1 对数色散）、等效阻抗片（胶层/薄片），以及 Redheffer 星积级联。提供可编程构建器与 JSON 配置驱动的全流程脚本。

## 功能特点
- 物理块（pressure-normalized）
  - 界面二端口：`S_interface(ZL, ZR)`
  - 均匀层：`S_layer(omega, c_p, d, ...)`
    - 非因果幅度衰减（给定 α(ω)）
    - 因果幂律衰减与色散：`0 < n < 2`，含 `n=1` 对数色散
  - 等效阻抗片（胶层/薄片）：`S_impedance_sheet(...)`，含面密度、粘弹刚度、频率相关损耗
- 级联：`star` 与 `fold_star`（数值正则，近极点稳定）
- 入端等效反射与透射：`gamma_in_from_S`、`Teff_from_S`
- 功率系数：`compute_RTA`（R/T/A，端口功率修正）
- 结构构建：`StructureBuilder`（链式 API），含胶层自动建模（sheet vs. explicit by |k d|）
- 配置驱动：`Builder_config` 将 `config.json` 翻译为构建与求解流程
- 测试脚本：一键跑通、保存 CSV/NPZ 与图像

## 目录结构
```
Multilayer_Model/
  multilayer_smatrix.py        # 物理内核与数值拼接
  Structure_builder.py         # StructureBuilder：可编程构建器与胶层助手
  Builder_config.py            # JSON 配置 → 构建与求解
  config.json                  # 示例配置
  test/
    run_config_flow.py         # 读取 config.json，输出 Γ_in 与图表
    run_config_flow_baseline.py# 含 R/T/A 与 baseline 归一
  artifacts/                   # 结果数据（CSV/NPZ）
  pictures/                    # 图像输出（PNG）
  README.md
```

## 架构与数据流
- 物理内核（`multilayer_smatrix.py`）
  - 只关心“块”与“拼接”，不关心结构含义或数据来源
  - 面向向量化频率轴 `omega`，广播友好
- 构建器（`Structure_builder.py`）
  - 对外提供链式 API：`add_interface`、`add_layer`、`add_adhesive_sheet/explicit/auto`、`build`、`gamma_in`
  - 胶层 auto：计算 `max |k d|` 与阈值 `kd_thresh` 比较，自动选择 sheet 或 explicit
- 配置胶水（`Builder_config.py`）
  - 读取 `config.json` 中 `frequency / media / layers / adhesives / chain`
  - 按链顺序组装，返回 `{omega, H_ref, S_tot}`，其中 `H_ref = Γ_in(ω)`（右端为半空间时）

数据流：`config.json` → `build_from_config` → `StructureBuilder`（内部调用物理函数）→ `S_tot` → `Γ_in(ω)` / `T_eff(ω)`

## 数学与模型要点
- 界面（压力归一）：`r = (Zb − Za)/(Zb + Za)`，`tab = 2 Zb/(Zb + Za)`，`tba = 2 Za/(Zb + Za)`
- 均匀层：`E(ω) = exp(−j k(ω) d)`
  - 非因果：`k = ω/c_p − j α(ω)`（仅幅度衰减）
  - 因果幂律：`k = ω/c0 + Δβ(ω) − j α(ω)`；`α(ω) = α0 (ω/ω0)^n`
    - `n ≠ 1`: `Δβ = α · cot(π n/2)`；`n = 1`: `Δβ = (2 α0/π) ω ln(ω/ω_ref)`
- 阻抗片（串联阻抗）：`Z_sheet(ω) = j ω m′ + R′(ω) + K*(ω)/(jω)`，`K*(ω)=K_n (1+j tanδ)` 或自定义 `K_n(ω)`
- 星积（Redheffer）：稳健正则 `eps_rel`，避免近极点失稳
- 功率修正：`T_power = |S21|^2 · Re{ZL}/Re{ZR}`；`A = 1 − R − T`（截断为非负）

## 快速开始（Windows 命令）
- 方式 A：配置驱动（推荐入门）
```bat
python test\run_config_flow.py
python test\run_config_flow_baseline.py
```
输出：
- `artifacts/run_config_*.csv|npz`、`test/artifacts/run_config_with_baseline_*.csv|npz`
- `pictures/run_config_*.png` 与 `test/pictures/run_config_with_baseline_*.png`

- 方式 B：可编程构建器 API
```python
import numpy as np
from Structure_builder import StructureBuilder
from Builder_config import load_json

# 频率轴
f = np.arange(0.05e6, 2.0e6 + 1e3, 1e3)
omega = 2*np.pi*f

# 介质阻抗示例
Z_water = 1000.0 * 1480.0
Z_rubber = 1100.0 * 1600.0
Z_metal  = 2780.0 * 6320.0

sb = StructureBuilder(omega)
(sb
 .add_interface(Z_water, Z_rubber)
 .add_layer(c_p=1600.0, d=1e-3, causal=False)
 .add_adhesive_auto(Z_rubber, Z_metal,
                    thickness=80e-6, rho=1200.0,
                    E_storage=2.0e9, tan_delta=0.08,
                    c_p=2000.0,  # 若省略 c_p 则强制 sheet
                    alpha0=40.0, n=1.3, f0_Hz=1.0e6,
                    kd_thresh=0.1)
)
S_tot = sb.build()
Gamma_in = sb.gamma_in(S_tot)
```

## 配置文件说明（`config.json`）
- `frequency`：
  - `start_Hz, stop_Hz, step_Hz` 或直接提供 `omega_rad_s`
- `media`：命名介质，含 `rho, c_p`；阻抗 `Z = ρ c_p`
- `layers`：有限厚度层
  - `name, medium, thickness_m`
  - `atten`（可选）：`alpha0_Np_per_m, n, f0_Hz`
    - 若 `0 < n < 2` 且 `n ≠ 1`，走因果幂律；否则退回非因果幅度衰减（内部按幂律生成 α(ω)）
- `adhesives`：胶层/薄片（按名字引用）
  - `mode: sheet | explicit | auto`
  - sheet：`rho, thickness_m, E_storage_Pa, tan_delta?, Rprime_Pa_s_per_m?`
  - explicit：`rho, thickness_m, c_p, atten{alpha0_Np_per_m, n, f0_Hz}`
  - auto：以上参数的组合；若提供 `c_p`，则根据 `max|k d|` 与 `kd_thresh` 决定 sheet/explicit
- `chain`：左→右顺序的构建列表，元素类型：
  - `interface {left, right}`：连接左右介质名
  - `layer {ref}`：引用 `layers[].name`
  - `adhesive {ref, left, right}`：引用 `adhesives` 名字，并指明左右端口介质
  - `halfspace {medium}`：右端半空间，视作 Γ_L=0 终端
- `baseline_override`（可选）：指定入射基线界面 `{left, right}` 用于基线归一

示例配置请见仓库 `config.json`。

## 输出与评估指标
- Γ_in：入端等效反射（右端半空间 ⇒ Γ_L=0）
- |Γ_in|、相位（解包）、回波损耗 RL(dB)、驻波比 SWR
- R/T/A：反射/透射/吸收功率系数（修正端口功率）
- 基线归一：
  - τ_base：入口界面 `4 ZL ZR / (ZL+ZR)^2`
  - η_down = (1 − R) / τ_base，η_T = T / τ_base
- 能量核对（无耗近似）：最大残差打印于 baseline 脚本输出

## 常见问题与提示
- 数值稳定：星积与端口求逆均使用 `eps_rel` 正则，避免近极点不适定
- 因果幂律：`0 < n < 2`，`n = 1` 使用对数色散，需参考频率 `ω_ref`（代码中默认取频带中位数）
- 胶层自动：建议阈值 `kd_thresh≈0.1`；`c_p` 未给出时强制 sheet 等效
- 端口功率：T 的功率修正因子为 `Re{ZL}/Re{ZR}`（压力归一）
- 运行环境：Python ≥ 3.7，NumPy；可选 Matplotlib 用于画图

## 运行与复现
- 安装依赖（最小）：
```bat
python -m pip install numpy matplotlib
```
- 运行配置流程：
```bat
python test\run_config_flow.py
python test\run_config_flow_baseline.py
```
- 结果位置：
  - 数据：`artifacts/`、`test/artifacts/`
  - 图像：`pictures/`、`test/pictures/`

## 变更记录（本次更新）
- 完善 README（架构、配置、使用、输出、稳定性）
- `test/run_config_flow.py` 修复运行路径导入问题、提升兼容性
- `test/three_layers test.py` 去除无效导入，补充本地数据类
- 细化与清理：去除未使用导入、类型注解兼容（Optional 替代 `|`）

如需将该项目嵌入参数扫描/优化/训练流程，建议优先使用 `StructureBuilder` 的链式 API；如需面向工程配置切换与复现实验环境，优先使用 `config.json + Builder_config` 路线。
