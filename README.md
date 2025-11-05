1) multilayer_smatrix.py（底层物理内核）

职责：只做物理块与数值拼接：

二端口块：S_interface、S_layer（含因果幂律衰减与 n=1 对数色散）、S_impedance_sheet（等效片/胶层）。

级联：star（Redheffer 星积，含小正则）、fold_star。

端口反射：gamma_in_from_S。

约束：不关心“结构长什么样”或“数据从哪里来”。接口稳定、向量化良好。

2) builder.py（StructureBuilder：可编程构建器）

职责：在不重写物理的前提下，提供链式 API 来组装结构：

add_interface(ZL, ZR)、add_layer(c_p, d, causal/atten)、add_adhesive_sheet/explicit/auto(...)。

build() 返回整链 S_tot；gamma_in(S_tot) 给出 

H(ω)=Γin(ω)。

胶层 auto：按max∣kd∣≤kd_thresh 在等效片/显式层之间切换。

约束：只依赖 multilayer_smatrix.py 的函数；不处理文件/配置解析；面向“写测试/扫参”的可编程场景。

3) builder_config.py（配置→构建胶水层）

职责：把JSON 配置翻译成 StructureBuilder 调用序列：

读取 frequency / media / layers / adhesives / chain。

按 chain 顺序逐项调用 StructureBuilder 的接口，最后输出 {omega, H_ref, S_tot}。

约束：不做物理、不做性能加速；确保配置解析与基本校验即可。

数据流：config.json → builder_config.build_from_config → StructureBuilder(调用 multilayer_smatrix) → S_tot → H(ω)
边界明确后，调参、换结构只改 JSON；写脚本/训练时只碰 builder.py 的 API。
