import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
md = 'cplxmd'
def _load_depth_vs_txt(txt_path):
    data = np.loadtxt(txt_path, skiprows=1)
    depth = data[:, 0]
    vs = data[:, 1]
    return depth, vs


def v2v(vp):
    rho = (
        1.6612 * vp
        - 0.4721 * vp**2
        + 0.0671 * vp**3
        - 0.0043 * vp**4
        + 0.000106 * vp**5
    )
    vs = (
        0.7858
        - 1.2344 * vp
        + 0.7949 * vp**2
        - 0.1238 * vp**3
        + 0.0064 * vp**4
    )
    return vs, rho


def bottom_layer_model_to_step(depth_bottom, velocity, z_top=0.0):
    """
    depth_bottom: 每层底界深度
    velocity:     每层速度

    例如：
    depth_bottom = [0.5, 1.5, 2.5, ..., 45, 80]
    velocity     = [v1,  v2,  v3,  ..., v10, v11]

    表示：
    0-0.5   : v1
    0.5-1.5 : v2
    ...
    45-80   : v11
    """
    depth_bottom = np.asarray(depth_bottom, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    z_edges = np.r_[z_top, depth_bottom]
    z_plot = np.repeat(z_edges, 2)[1:-1]
    v_plot = np.repeat(velocity, 2)

    return z_plot, v_plot


# -------------------------
# 读取数据
# -------------------------
dp_in, vs_inv = _load_depth_vs_txt(f"/home/jak123/work/tarim/check/md1/invrf/results/{md}/csv/posterior_vs_median.txt")

data = np.loadtxt(f"/home/jak123/work/tarim/check/{md}/md1.txt")
dp_tr = data[:, 0]
vp_tr = data[:, 1]
vs_tr, _ = v2v(vp_tr)

# true layered model -> step curve
z_true_plot, vs_true_plot = bottom_layer_model_to_step(dp_tr, vs_tr, z_top=0.0)

# -------------------------
# 绘图
# -------------------------
fig, ax = plt.subplots(figsize=(5.2, 7.2), dpi=200)

# 白底+更清爽的边框
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
for spine in ax.spines.values():
    spine.set_linewidth(1.1)

# 反演模型：连续、醒目
ax.plot(
    vs_inv, dp_in,
    lw=2.8,
    color="#1f77b4",
    label="Inv model (median)"
)

# 真实模型：阶梯、稍细一些
ax.plot(
    vs_true_plot, z_true_plot,
    lw=2.2,
    color="#ff7f0e",
    ls=(0, (6, 3)),   # 自定义虚线，更好看
    label="True model"
)

# 在层界面位置加小横线（可选，美观也更清楚）
for d, v in zip(dp_tr, vs_tr):
    ax.plot(v, d, marker="_", ms=10, mew=1.4, color="#ff7f0e")

# 坐标轴设置
ax.invert_yaxis()
ax.set_ylim(dp_tr.max() + 2, 0)
ax.set_xlim(0.6, max(vs_inv.max(), vs_true_plot.max()) + 0.25)

ax.set_xlabel("Vs (km/s)", fontsize=15)
ax.set_ylabel("Depth (km)", fontsize=15)
ax.set_title("Vs model comparison", fontsize=18, pad=12)

# 刻度
ax.set_yticks(np.arange(0, 81, 10))
ax.tick_params(axis="both", which="major", labelsize=12, length=5, width=1.0)

# 浅色网格
ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.9, alpha=0.35)

# 图例：小一点，不遮挡主体
leg = ax.legend(
    loc="lower left",
    fontsize=12,
    frameon=True,
    framealpha=0.9,
    edgecolor="0.8"
)
leg.get_frame().set_facecolor("white")

plt.tight_layout()
plt.savefig(f"{md}_vs_c.png", dpi=300, bbox_inches="tight")
plt.close()
print("[OK] saved figure: vs_model_comparison_pretty.png")

