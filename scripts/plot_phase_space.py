import numpy as np
from skimage.measure import EllipseModel, ransac
from matplotlib.patches import Ellipse
import tfs
from aba_optimiser.config import TRACK_DATA_FILE, BPM_RANGE, RAMP_UP_TURNS, NOISE_FILE
from aba_optimiser.utils import select_marker
import matplotlib.pyplot as plt


def compute_twiss(a, b, theta):
    """
    Given ellipse semi-axes a, b and rotation theta (rad),
    return beta, alpha, gamma, eps.
    """
    # Rotation matrix
    R = np.array([[ np.cos(theta), -np.sin(theta)],
                  [ np.sin(theta),  np.cos(theta)]])
    # Conic matrix Q such that (X-mu)^T Q (X-mu) = 1
    D = np.diag([1/a**2, 1/b**2])
    Q = R @ D @ R.T

    # Covariance = Sigma = Q^{-1}
    Sigma = np.linalg.inv(Q)

    # emittance = sqrt(det Sigma)
    eps = np.sqrt(np.linalg.det(Sigma))

    beta  = Sigma[0,0] / eps
    alpha = -Sigma[0,1] / eps
    gamma = Sigma[1,1] / eps

    return beta, alpha, gamma, eps

def fit_true_ellipse(x, y, tol, max_trials=1000):
    """
    Robustly fit an ellipse to (x,y) via RANSAC+EllipseModel.
    tol: residual threshold in same units as x,y
    """
    pts = np.column_stack([x, y])
    model_robust, inliers = ransac(
        (pts,),                  # data
        EllipseModel,            # model to fit
        min_samples=5,           # #points to fit a candidate
        residual_threshold=tol,  # how far a point can be from the ellipse
        max_trials=max_trials
    )
    return model_robust.params  # (xc, yc, a, b, theta)

def plot_recovered_ellipse(ax, x, y, tol, **ellipse_kwargs):
    xc, yc, a, b, theta = fit_true_ellipse(x, y, tol)
    ell = Ellipse(
        (xc, yc),
        width = 2*a,
        height = 2*b,
        angle = np.degrees(theta),
        facecolor = 'none',
        **ellipse_kwargs
    )
    ax.add_patch(ell)

    beta, alpha, gamma, eps = compute_twiss(a, b, theta)
    print(f"β = {beta:.3f},  α = {alpha:.3f},  γ = {gamma:.3f},  ε = {eps:.3e}")

    return ell

tol_xpx = 1e-4
tol_ypy = 1e-4
# Read non-noisy data (TRACK_DATA_FILE)
init_coords = tfs.read(TRACK_DATA_FILE, index="turn")
start_bpm, _ = BPM_RANGE.split("/")
other_bpm = "BPM.14R3.B1"  # Example of another BPM

non_noisy_start = select_marker(init_coords, start_bpm)
non_noisy_start = non_noisy_start[non_noisy_start.index > RAMP_UP_TURNS]

non_noisy_other = tfs.read(TRACK_DATA_FILE, index="turn")
non_noisy_other = select_marker(non_noisy_other, other_bpm)
non_noisy_other = non_noisy_other[non_noisy_other.index > RAMP_UP_TURNS]

# Read noisy data (NOISE_FILE)
noise_init = tfs.read(NOISE_FILE, index="turn")
noisy_start = select_marker(noise_init, start_bpm)
noisy_start = noisy_start[noisy_start.index > RAMP_UP_TURNS]

noise_other = tfs.read(NOISE_FILE, index="turn")
noise_other = select_marker(noise_other, other_bpm)
noise_other = noise_other[noise_other.index > RAMP_UP_TURNS]

# Create a 2x2 subplot for phase space plots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Subplot 0,0: x vs px for start BPM
axs[0, 0].scatter(noisy_start["x"], noisy_start["px"], s=1, color='blue', label='Noisy')
axs[0, 0].scatter(non_noisy_start["x"], non_noisy_start["px"], s=1, color='red', label='Non-noisy')
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("px")
axs[0, 0].set_title(f"x, px Phase Space ({start_bpm})")
axs[0, 0].grid()
# plot_recovered_ellipse(
#     axs[0,0],
#     noisy_start["x"], noisy_start["px"],
#     tol_xpx,
#     edgecolor='green', linewidth=2,
#     label='Recovered Ellipse'
# )

# Subplot 0,1: y vs py for start BPM
axs[0, 1].scatter(noisy_start["y"], noisy_start["py"], s=1, color='blue', label='Noisy')
axs[0, 1].scatter(non_noisy_start["y"], non_noisy_start["py"], s=1, color='red', label='Non-noisy')
axs[0, 1].set_xlabel("y")
axs[0, 1].set_ylabel("py")
axs[0, 1].set_title(f"y, py Phase Space ({start_bpm})")
axs[0, 1].grid()
# plot_recovered_ellipse(
#     axs[0, 1],
#     noisy_start["y"], noisy_start["py"],
#     tol_ypy,
#     edgecolor='green', linewidth=2,
#     label='Recovered Ellipse'
# )

# Subplot 1,0: x vs px for end BPM
axs[1, 0].scatter(noise_other["x"], noise_other["px"], s=1, color='blue', label='Noisy')
axs[1, 0].scatter(non_noisy_other["x"], non_noisy_other["px"], s=1, color='red', label='Non-noisy')
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("px")
axs[1, 0].set_title(f"x, px Phase Space ({other_bpm})")
axs[1, 0].grid()
# plot_recovered_ellipse(
#     axs[1, 0],
#     noise_other["x"], noise_other["px"],
#     tol_xpx,
#     edgecolor='green', linewidth=2,
#     label='Recovered Ellipse'
# )

# Subplot 1,1: y vs py for end BPM
axs[1, 1].scatter(noise_other["y"], noise_other["py"], s=1, color='blue', label='Noisy')
axs[1, 1].scatter(non_noisy_other["y"], non_noisy_other["py"], s=1, color='red', label='Non-noisy')
axs[1, 1].set_xlabel("y")
axs[1, 1].set_ylabel("py")
axs[1, 1].set_title(f"y, py Phase Space ({other_bpm})")
axs[1, 1].grid()
# plot_recovered_ellipse(
#     axs[1, 1],
#     noise_other["y"], noise_other["py"],
#     tol_ypy,
#     edgecolor='green', linewidth=2,
#     label='Recovered Ellipse'
# )

# Create one global legend for the entire figure
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1))

plt.suptitle("Phase Space Comparison", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("merged_phase_space_comparison.png", dpi=300)
plt.show()