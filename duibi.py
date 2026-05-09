import os

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

def _load_depth_vs_txt(txt_path):
    """
    读取两列 txt:
    depth_km vs_km_s
    """
    data = np.loadtxt(txt_path, skiprows=1)
    depth = data[:, 0]
    vs = data[:, 1]
    return depth, vs
def v2v(vp):
    #vp = 0.9409 + 2.0947*vs - 0.8206*vs**2 + 0.2683*vs**3 - 0.0251*vs**4
    rho = 1.6612*vp - 0.4721*vp**2 + 0.0671*vp**3 - 0.0043*vp**4 + 0.000106*vp**5
    vs = 0.7858 - 1.2344*vp + 0.7949*vp**2 - 0.1238*vp**3 + 0.0064*vp**4
#    k = vp/vs
    return vs, rho
def bottom_layer_model_to_step(depth_bottom, velocity, z_top=0.0):
    depth_bottom = np.asarray(depth_bottom, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    z_edges = np.r_[z_top, depth_bottom]

    z_plot = np.repeat(z_edges, 2)[1:-1]
    v_plot = np.repeat(velocity, 2)

    return z_plot, v_plot

dp_in ,vs_inv = _load_depth_vs_txt(f"results/{md}/csv/posterior_vs_median.txt")
data = np.loadtxt(f"../../{md}/md1.txt")
dp_tr = data[:,0]
vp_tr = data[:,1]
vs_tr,_ = v2v(vp_tr)
z_true_plot, vs_true_plot = bottom_layer_model_to_step(dp_tr, vs_tr, z_top=0.0)
fig, ax = plt.subplots(figsize=(4.5, 6))
ax.plot(
    vs_inv,
    dp_in,
    lw=2.5,
    label="Inv-model(median)"
)
ax.plot(
    vs_true_plot,
    z_true_plot,
    lw=2.5,
    ls="--",
    label="True-model"
)
ax.invert_yaxis()

ax.set_xlabel("Vs (km/s)")
ax.set_ylabel("Depth (km)")
ax.set_title("Vs model comparison")
ax.grid(True, ls=":", alpha=0.6)
ax.legend()

plt.tight_layout()
plt.savefig("vs_model_comparison.png", dpi=300)
plt.close()
