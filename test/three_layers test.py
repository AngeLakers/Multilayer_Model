import numpy as np
import matplotlib.pyplot as plt

from multilayer_smatrix import (
    S_interface, S_layer, fold_star, gamma_in_from_S,
    LayerParam, alpha_powerlaw
)

# --------------------- frequency axis ---------------------
f_start = 0.1e6   # 100 kHz
f_stop  = 3.0e6   # 3 MHz
df      = 1.0e3   # 1 kHz step
# 用 arange 生成，[start, stop] 端点包含与否由步长决定；这里手动含终点
f = np.arange(f_start, f_stop + 0.5*df, df)
omega = 2*np.pi*f
omega0 = 2*np.pi*((f_start + f_stop)/2)

# --------------------- materials (thickness in meters) ---------------------
# 半空间（水、基体）不建厚度；有限层给 d
water  = LayerParam(rho=1000, c_p=1480, d=0.0,   alpha0=0.0, n=1.0)   # 上方耦合水（半空间）

steel1 = LayerParam(rho=7850, c_p=5900, d=0.0005, alpha0=0.0, n=1.0)  # 钢 0.5 mm
rubber1= LayerParam(rho=1100, c_p=1600, d=0.00063,alpha0=0.0, n=1.2)  # 橡胶 0.63 mm

# 加热组件：显式薄层（NiCr 例）
heater = LayerParam(rho=8400, c_p=5600, d=0.0001, alpha0=0.0, n=1.0)  # NiCr 箔 0.1 mm

# “常用合金层”：示例用铝合金 2024-T3（你可改成 Ti-6Al-4V 等）
alloy  = LayerParam(rho=2780, c_p=6320, d=0.0010, alpha0=0.0, n=1.0)  # Al 2024 1.0 mm

rubber2= LayerParam(rho=1100, c_p=1600, d=0.0013, alpha0=0.0, n=1.2)  # 橡胶 1.3 mm

# 底部“机翼组件”视为半空间；先用铝合金的 ρc 作为例子（可改成 CFRP/Ti 等）
substrate_Z = 2780 * 6320   # 若要复合材料或钛，请替换为对应 ρ·c


Z = lambda p: p.rho * p.c_p
a = lambda p: alpha_powerlaw(omega, p.alpha0, p.n, omega0)

# --------------------- build S-block chain: Water | Steel1 | Rubber | Steel2(half-space) ---------------------
blocks = [S_interface(Z(water), Z(steel1)), S_layer(omega, steel1.c_p, steel1.d, a(steel1)),
          S_interface(Z(steel1), Z(rubber1)), S_layer(omega, rubber1.c_p, rubber1.d, a(rubber1)),
          S_interface(Z(rubber1), Z(heater)), S_layer(omega, heater.c_p, heater.d, a(heater)),
          S_interface(Z(heater), Z(alloy)), S_layer(omega, alloy.c_p, alloy.d, a(alloy)),
          S_interface(Z(alloy), Z(rubber2)), S_layer(omega, rubber2.c_p, rubber2.d, a(rubber2)),
          S_interface(Z(rubber2), substrate_Z)]
# 水 | 钢(0.5mm)

# 钢 | 橡胶(0.63mm)

# 橡胶 | 加热箔(0.1mm NiCr)

# 加热箔 | 合金层(Al 2024, 1.0mm)

# 合金层 | 橡胶(1.3mm)

# 橡胶 | 机翼组件半空间（终端）

# 折叠与输出
S_tot    = fold_star(blocks)
Gamma_in = gamma_in_from_S(S_tot, Gamma_L=None)
R        = np.abs(Gamma_in)**2
phi      = np.unwrap(np.angle(Gamma_in))
# --------------------- helpers ---------------------
def plot_with_auto_x(x_Hz, y, ylabel, title):
    x_MHz = x_Hz / 1e6
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(x_MHz, y)
    ax.set_xlim(x_MHz[0], x_MHz[-1])           # 轴范围由数据自动决定
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    plt.show()

# --------------------- plots (auto-scaled) ---------------------
plot_with_auto_x(f, np.abs(Gamma_in), '|Γ_in|',
                 'Water | Steel(1mm) | Rubber(1mm) | Steel (half-space)')
plot_with_auto_x(f, R, '|Γ_in|^2 (Reflection Power)',
                 'Reflection Power Spectrum')
plot_with_auto_x(f, phi, 'Phase (rad)',
                 'Phase of Γ_in (unwrapped)')

# --------------------- prints (auto from data) ---------------------
print(f"Freq points: {len(f)}")
print(f"Freq range: {f[0]/1e6:.3f} → {f[-1]/1e6:.3f} MHz   step: { (f[1]-f[0])/1e3:.0f} kHz")
print(f"max |Γ_in| = {np.max(np.abs(Gamma_in)):.4f},  min |Γ_in| = {np.min(np.abs(Gamma_in)):.4f}")
