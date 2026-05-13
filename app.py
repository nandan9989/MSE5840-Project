import os
import tempfile
import base64
from math import comb

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Needed so matplotlib works cleanly inside Streamlit
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.animation import FuncAnimation, PillowWriter

# ------------------------------------------------------------
# Optional SciPy import
# We use scipy only for optional alpha-stable Monte Carlo
# sampling of Lévy flights. The main theory plots do NOT depend
# on this and still work without SciPy.
# ------------------------------------------------------------
try:
    from scipy.stats import levy_stable
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ============================================================
# STREAMLIT PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Random Walk and Lévy Flight Theory App",
    layout="wide"
)

st.title("Random Walk and Lévy Flight Theory App")

st.markdown("""
## What is a Lévy Flight?

Imagine you are tracking an animal foraging for food. You will find that most of the time it takes small, nearby steps, enough to thoroughly search an area before making a sudden, very long jump to an entirely different area. That combination of many small steps punctuated by rare large ones is the defining feature of a **Lévy flight**.

Standard random walks are well-documented. Every step has a fixed length, but random direction, and over time the object drifts away from the origin with a characteristic spread that grows as the square root of the number of steps. This is known as **normal diffusion**, which produces a distribution of positions in the shape of a Gaussian.

A Lévy flight replaces those fixed-size steps with step lengths drawn from a heavy-tailed distribution, where the extreme events are far more likely than they would be in a Gaussian distribution. Power-law decay in Lévy flights as opposed to exponential decay for Gaussians is the fundamental difference here. The result is known as **superdiffusion**, where the object spreads through space more quickly than in a standard random walk, even when both methods are normalized to the average step length.

Lévy flights appear in a surprising range of real systems: beyond just the foraging paths of albatrosses and sharks, they can also describe things like cell motion and even fluctuations in financial markets. The key is that in processes where the rare, large events play an outsized role, a Lévy flight can be a more accurate model than a standard random walk.

The key parameter controlling everything is **μ** (mu):
- **μ = 2** recovers normal Gaussian diffusion — the Lévy flight becomes an ordinary random walk.
- **μ < 2** produces superdiffusion with increasingly heavy tails as μ decreases toward 0.
- The smaller μ is, the more "jumpy" the flight becomes, with more frequent large leaps.

---

## Key Equations

### Standard Random Walk

In the long-time (diffusion) limit, the probability of finding the walker at position $x$ at time $t$ is a Gaussian:
""")

st.latex(r"p(x, t) = \frac{1}{\sqrt{4\pi D t}} \exp\!\left(-\frac{x^2}{4Dt}\right)")

st.markdown(r"where $D = a^2 \Gamma / 2$ is the diffusion coefficient. The mean squared displacement (MSD) grows linearly in time — the hallmark of normal diffusion:")

st.latex(r"\langle x^2 \rangle = 2Dt")

st.markdown("""
### Lévy Flight

A Lévy flight is characterized by its **characteristic function** — the Fourier transform of the position distribution:
""")

st.latex(r"\hat{p}(k, t) = e^{-D_1 |k|^\mu t}, \qquad 0 < \mu \leq 2")

st.markdown("This produces a distribution with **power-law tails** instead of exponential decay:")

st.latex(r"P(x, t) \sim |x|^{-(1+\mu)} \quad \text{for large } |x|")

st.markdown(r"Because of these heavy tails, the variance diverges for $\mu < 2$, so MSD is no longer a useful measure of spread. Instead, the natural measure is the width of the distribution (e.g. the interquartile range), which scales as:")

st.latex(r"\text{width} \sim t^{1/\mu}")

st.markdown(r"Since $1/\mu > 1/2$ whenever $\mu < 2$, a Lévy walker always spreads faster than a standard random walker. The full distribution also satisfies a **scaling collapse** — its shape at any time is the same function, just stretched:")

st.latex(r"P(x, t) = t^{-1/\mu} \, G\!\left(\frac{x}{t^{1/\mu}}\right)")

st.markdown(r"Setting $\mu = 2$ recovers the Gaussian exactly, with $D_1 = D$.")

st.markdown("""
---

## Parameter Guide

| Parameter | Symbol | What it controls |
|---|---|---|
| **Step length** | *a* | The fixed step size used in the standard random walk. |
| **Jump frequency** | *Γ* | How many steps the walker takes per unit time. Together with *t*, this sets the total number of steps *N = Γt*. |
| **Time** | *t* | The total time elapsed in the simulation. |
| **Monte Carlo walkers** | — | How many independent walkers to simulate for the histograms. More walkers give smoother distributions but take longer. |
| **μ (mu)** | *μ* | The stability index of the Lévy flight. Controls how heavy the tails are. Must be between 0 and 2. |
| **D₁** | *D₁* | The generalized diffusion coefficient for the Lévy flight. Scales the overall spread without changing the shape. |
| **Times** | — | A comma-separated list of time values at which to plot the Lévy PDF. Use a range spanning an order of magnitude or more to see the scaling behavior clearly. |
| **x max / Grid points** | — | The spatial range and resolution of the numerical PDF grid. Increase grid points for sharper plots; increase x max if the PDF appears cut off at the edges. |
| **Animated paths** | — | The number of individual walker trajectories shown in the animations. |
| **Animation steps** | — | How many steps each animated walker takes. More steps show longer-term behavior. |
| **Animation dt** | *dt* | The time increment per step in the Lévy animation. Scales the step-length distribution. |

---
""")
st.write("Adjust the parameters in the sidebar and press **Run** to generate the animations and theory plots.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def ensure_positive_integer_close(x, name):
    """
    Some quantities, such as N = Γ t, must be integer-like for the
    discrete random walk model. This checks that x is very close to
    an integer and returns that integer.
    """
    xr = int(round(x))
    if abs(x - xr) > 1e-12 or xr < 1:
        raise ValueError(
            f"{name} must be a positive integer-compatible value. Got {x}"
        )
    return xr


def anim_to_base64_gif(anim):
    """
    Convert a matplotlib animation into a base64 GIF string so it can
    be displayed directly in Streamlit with an HTML <img> tag.

    Important:
    PillowWriter expects a real file path, not a BytesIO object.
    So we save temporarily to a .gif file, read it back, then delete it.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmp:
        temp_path = tmp.name

    try:
        writer = PillowWriter(fps=12)
        anim.save(temp_path, writer=writer)

        with open(temp_path, "rb") as f:
            data = f.read()

        return base64.b64encode(data).decode("utf-8")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# RANDOM WALK THEORY + MONTE CARLO
# ============================================================
def rw_exact_distribution(N, a):
    """
    Exact final-position distribution for a 1D lattice random walk.

    After N steps, if each step is ±a with equal probability,
    the final position is:
        x = (2m - N) a
    where m is the number of +a steps.

    The corresponding probability is binomial:
        P(m) = C(N, m) / 2^N
    """
    m = np.arange(N + 1)
    x = (2 * m - N) * a
    p = np.array([comb(N, int(mi)) for mi in m], dtype=np.float64) / (2 ** N)
    return x, p


def rw_moments(N, a, Gamma, t):
    """
    Standard random-walk theory:
        <R>   = 0
        <R^2> = N a^2

    With N = Γ t and in 1D:
        D = a^2 Γ / 2
        <x^2> = 2 D t
    """
    mean = 0.0
    msd = N * a**2
    D = 0.5 * a**2 * Gamma
    msd_diff = 2.0 * D * t
    return mean, msd, D, msd_diff


def rw_mc_final_positions(num_walkers, N, a, seed):
    """
    Monte Carlo simulation of final positions for a 1D random walk.
    Each step is +a or -a with equal probability.
    """
    rng = np.random.default_rng(seed)
    steps = rng.choice([-a, a], size=(num_walkers, N))
    return steps.sum(axis=1)


def rw_mc_trajectories_2d(num_paths, num_steps, a, seed):
    """
    2D trajectory visualization for random walks.

    This is used for animations. Each step has fixed length a and a
    random direction. This is visually useful for comparing spreading
    with Lévy flights.
    """
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, size=(num_paths, num_steps))

    dx = a * np.cos(angles)
    dy = a * np.sin(angles)

    x = np.zeros((num_paths, num_steps + 1))
    y = np.zeros((num_paths, num_steps + 1))
    x[:, 1:] = np.cumsum(dx, axis=1)
    y[:, 1:] = np.cumsum(dy, axis=1)

    return x, y


def gaussian_pdf(x, D, t):
    """
    Diffusion-limit Gaussian density in 1D:
        p(x,t) = 1 / sqrt(4πDt) * exp[-x^2 / (4Dt)]
    """
    return 1.0 / np.sqrt(4.0 * np.pi * D * t) * np.exp(-(x**2) / (4.0 * D * t))


# ============================================================
# LÉVY THEORY + OPTIONAL MONTE CARLO
# ============================================================
def levy_pdf_fourier(x_grid, t, mu, D1):
    """
    Compute the pure-case Lévy-flight density by numerically inverting
    the characteristic function:

        P(x,t) = (1 / 2π) ∫ exp(i k x - D1 |k|^μ t) dk

    This is done using FFT methods.

    Notes:
    - mu controls the tail heaviness
    - for mu < 2, the process is non-Gaussian
    - the ideal variance diverges for mu < 2
    """
    n = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    L = n * dx

    # Wavenumber grid
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)

    # Characteristic function in Fourier space
    phi = np.exp(-D1 * (np.abs(k) ** mu) * t)

    # Numerical inverse Fourier transform
    dk = 2.0 * np.pi / L
    P = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(phi)))
    P = np.real(P) * n * dk / (2.0 * np.pi)
    P = np.fft.fftshift(P)

    # Small numerical negatives can appear from roundoff
    P[P < 0] = 0.0

    # Normalize to unit area
    area = np.trapz(P, x_grid)
    if area > 0:
        P /= area

    return P


def levy_scaling_rescale(x, p, t, mu):
    """
    Rescale the Lévy PDF according to the scaling form:

        P(x,t) = t^(-1/mu) G(x / t^(1/mu))

    Therefore, for scaling collapse we plot:
        X = x / t^(1/mu)
        Y = t^(1/mu) P(x,t)
    """
    X = x / (t ** (1.0 / mu))
    Y = (t ** (1.0 / mu)) * p
    return X, Y


def levy_quantiles_from_pdf(x, p, probs=(0.25, 0.5, 0.75)):
    """
    Compute selected quantiles from a numerically sampled PDF.
    We use these for width-based scaling instead of MSD,
    because for mu < 2 the ideal variance diverges.
    """
    cdf = np.cumsum(p)
    cdf = cdf / cdf[-1]

    out = {}
    for q in probs:
        idx = np.searchsorted(cdf, q)
        idx = min(max(idx, 0), len(x) - 1)
        out[q] = x[idx]
    return out


def levy_width_slope(times, x_grid, pdfs):
    """
    Use interquartile width as a robust finite measure of spread.
    The width should scale like:
        width ~ t^(1/mu)

    We estimate the slope in log-log space.
    """
    widths = []
    for p in pdfs:
        qs = levy_quantiles_from_pdf(x_grid, p, probs=(0.25, 0.75))
        widths.append(qs[0.75] - qs[0.25])

    times = np.array(times, dtype=float)
    widths = np.array(widths, dtype=float)

    slope, intercept = np.polyfit(np.log(times), np.log(widths), 1)
    return widths, slope, intercept


def levy_mc_samples(mu, D1, t, n, seed):
    """
    Optional Monte Carlo sample generation for 1D Lévy stable variables.
    This is only used for a visual theory-vs-MC check.

    The characteristic function for scipy's symmetric stable law is:
        exp(-scale^mu |k|^mu)

    We want:
        exp(-D1 t |k|^mu)

    Therefore:
        scale = (D1 t)^(1/mu)
    """
    if not HAS_SCIPY:
        return None

    rng = np.random.default_rng(seed)
    scale = (D1 * t) ** (1.0 / mu)

    return levy_stable.rvs(
        alpha=mu,
        beta=0.0,
        loc=0.0,
        scale=scale,
        size=n,
        random_state=rng
    )


def levy_mc_trajectories_2d(num_paths, num_steps, mu, D1, dt, seed):
    """
    2D trajectory visualization for Lévy flights.

    For each time step:
    - draw a step length from a heavy-tailed alpha-stable law
    - choose a random direction
    - update the position

    This is mainly for animation and visual comparison.
    """
    rng = np.random.default_rng(seed)

    if HAS_SCIPY:
        scale = (D1 * dt) ** (1.0 / mu)
        lengths = np.abs(
            levy_stable.rvs(
                alpha=mu,
                beta=0.0,
                loc=0.0,
                scale=scale,
                size=(num_paths, num_steps),
                random_state=rng
            )
        )
    else:
        # Heavy-tail fallback if SciPy is unavailable
        lengths = (D1 * dt) ** (1.0 / mu) * (rng.pareto(mu, size=(num_paths, num_steps)) + 1.0)

    angles = rng.uniform(0, 2 * np.pi, size=(num_paths, num_steps))
    dx = lengths * np.cos(angles)
    dy = lengths * np.sin(angles)

    x = np.zeros((num_paths, num_steps + 1))
    y = np.zeros((num_paths, num_steps + 1))
    x[:, 1:] = np.cumsum(dx, axis=1)
    y[:, 1:] = np.cumsum(dy, axis=1)

    return x, y


# ============================================================
# STATIC PLOTTING FUNCTIONS
# ============================================================
def plot_rw_distribution(N, a, D, t, finals):
    """
    Show:
    - exact discrete distribution
    - Monte Carlo histogram
    - Gaussian diffusion limit
    """
    x_exact, p_exact = rw_exact_distribution(N, a)
    x_dense = np.linspace(x_exact.min() - 5 * a, x_exact.max() + 5 * a, 2000)
    p_gauss = gaussian_pdf(x_dense, D, t)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stem(
        x_exact, p_exact,
        basefmt=" ",
        linefmt="C0-",
        markerfmt="C0o",
        label="Exact distribution"
    )
    ax.hist(finals, bins=60, density=True, alpha=0.35, label="Monte Carlo")
    ax.plot(x_dense, p_gauss, linewidth=2.5, label="Gaussian limit")

    ax.set_xlabel("x")
    ax.set_ylabel("Probability / density")
    ax.set_title("Random Walk Distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_rw_msd_check(N_values, a, Gamma, num_walkers, seed):
    """
    Compare theoretical <R^2> = N a^2 against Monte Carlo estimates
    for several values of N.
    """
    th_vals = []
    mc_vals = []

    for N in N_values:
        finals = rw_mc_final_positions(min(num_walkers, 20000), int(N), a, seed + int(N))
        th_vals.append(N * a**2)
        mc_vals.append(np.mean(finals**2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N_values, th_vals, "o-", linewidth=2.5, label="Theory")
    ax.plot(N_values, mc_vals, "s--", linewidth=2.0, label="Monte Carlo")

    ax.set_xlabel("N")
    ax.set_ylabel("<R²>")
    ax.set_title("Random Walk Mean-Square Check")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_levy_pdfs(x_grid, times, pdfs):
    """
    Plot Lévy PDFs for several times.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for t, p in zip(times, pdfs):
        ax.plot(x_grid, p, linewidth=2.2, label=f"t={t}")

    ax.set_xlabel("x")
    ax.set_ylabel("P(x,t)")
    ax.set_title("Lévy PDFs")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_levy_scaling(x_grid, times, pdfs, mu):
    """
    Plot rescaled curves to verify scaling collapse.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for t, p in zip(times, pdfs):
        X, Y = levy_scaling_rescale(x_grid, p, t, mu)
        ax.plot(X, Y, linewidth=2.2, label=f"t={t}")

    ax.set_xlabel("x / t^(1/μ)")
    ax.set_ylabel("t^(1/μ) P(x,t)")
    ax.set_title("Scaling Collapse")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_levy_width(times, widths, slope, intercept, mu):
    """
    Plot width scaling in log-log coordinates.
    """
    times = np.array(times, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(times, widths, "o-", linewidth=2.2, label="Numerical width")
    ax.loglog(
        times,
        np.exp(intercept) * times**(1.0 / mu),
        "--",
        linewidth=2.2,
        label=f"Reference slope 1/μ = {1.0 / mu:.4f}"
    )

    ax.set_xlabel("t")
    ax.set_ylabel("IQR width")
    ax.set_title(f"Width Scaling (estimated slope = {slope:.4f})")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_levy_mc_vs_theory(x_grid, p_theory, p_gaussian, levy_samples, mu, t):
    """
    Compare Lévy Monte Carlo histogram against both the Lévy and Gaussian
    theoretical PDFs on the same axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    levy_clipped = levy_samples[np.abs(levy_samples) < 100]
    ax.hist(levy_clipped, bins=300, density=True, alpha=0.35, label="Lévy Monte Carlo")
    ax.plot(x_grid, p_theory, linewidth=2.5, label=f"Lévy theory (μ={mu})")
    ax.plot(x_grid, p_gaussian, linewidth=2.5, linestyle="--", label="Gaussian theory")

    ax.set_xlim(-50, 50)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title(f"Lévy vs Gaussian Distribution (t={t})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


# ============================================================
# ANIMATION FUNCTIONS
# ============================================================
def make_path_animation(x, y, title):
    """
    Animate full trajectories of multiple particles.
    """
    n_paths, n_frames = x.shape
    fig, ax = plt.subplots(figsize=(6, 6))

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    pad_x = 0.05 * (xmax - xmin + 1e-9)
    pad_y = 0.05 * (ymax - ymin + 1e-9)

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)

    lines = [ax.plot([], [], linewidth=1.5)[0] for _ in range(n_paths)]
    dots = [ax.plot([], [], "o", markersize=4)[0] for _ in range(n_paths)]

    def init():
        for line, dot in zip(lines, dots):
            line.set_data([], [])
            dot.set_data([], [])
        return lines + dots

    def update(frame):
        for i in range(n_paths):
            lines[i].set_data(x[i, :frame + 1], y[i, :frame + 1])
            dots[i].set_data([x[i, frame]], [y[i, frame]])
        return lines + dots

    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        init_func=init,
        interval=70,
        blit=False
    )
    return anim, fig


def make_cloud_animation(x, y, title):
    """
    Animate the particle cloud only, without drawing full trails.
    """
    n_paths, n_frames = x.shape
    fig, ax = plt.subplots(figsize=(6, 6))

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    pad_x = 0.05 * (xmax - xmin + 1e-9)
    pad_y = 0.05 * (ymax - ymin + 1e-9)

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)

    scat = ax.scatter([], [], s=18, alpha=0.7)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return (scat,)

    def update(frame):
        pts = np.column_stack((x[:, frame], y[:, frame]))
        scat.set_offsets(pts)
        return (scat,)

    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        init_func=init,
        interval=70,
        blit=False
    )
    return anim, fig


# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.header("Parameters")

if "default_seed" not in st.session_state:
    st.session_state.default_seed = int(np.random.randint(1, 101))
seed = st.sidebar.number_input("Random seed", value=st.session_state.default_seed, step=1)

st.sidebar.subheader("Random walk")
a = st.sidebar.number_input("Step length a", value=1.0)
Gamma = st.sidebar.number_input("Jump frequency Γ", value=1.0)
t_rw = st.sidebar.number_input("Time t", value=120.0)
num_walkers = st.sidebar.number_input("Monte Carlo walkers", value=20000, step=1000)

st.sidebar.subheader("Lévy")
mu = st.sidebar.slider("μ", 0.5, 2.0, 1.5, 0.1)
D1 = st.sidebar.number_input("D₁", value=1.0)
times_text = st.sidebar.text_input("Times", "1,4,16,64")
xmax = st.sidebar.number_input("x max", value=80.0)
nx = st.sidebar.number_input("Grid points", value=4096, step=256)

st.sidebar.subheader("Animations")
num_anim_paths = st.sidebar.slider("Animated paths", 3, 30, 8)
num_anim_steps = st.sidebar.slider("Animation steps", 20, 200, 80)
dt_anim = st.sidebar.number_input("Animation dt", value=1.0)

run = st.sidebar.button("Run")


# ============================================================
# MAIN APP LOGIC
# ============================================================
if run:
    try:
        # ----------------------------------------------------
        # Validate and parse inputs
        # ----------------------------------------------------
        N = ensure_positive_integer_close(Gamma * t_rw, "Γt")
        times = [float(v.strip()) for v in times_text.split(",") if v.strip()]

        if mu <= 0 or mu > 2:
            raise ValueError("μ must satisfy 0 < μ ≤ 2")

        if a <= 0 or Gamma <= 0 or t_rw <= 0 or D1 <= 0 or xmax <= 0 or nx < 256:
            raise ValueError("Use positive parameters, and grid points should be at least 256.")

        # ----------------------------------------------------
        # Animation section (shown first as the primary visual)
        # ----------------------------------------------------
        st.markdown('<div id="anim-anchor"></div>', unsafe_allow_html=True)
        components.html(
            "<script>"
            "setTimeout(function(){"
            "  var el = window.parent.document.getElementById('anim-anchor');"
            "  if (el) el.scrollIntoView({behavior: 'smooth'});"
            "}, 100);"
            "</script>",
            height=0
        )
        st.subheader("Animations")

        with st.spinner("Generating animations..."):
            rwx, rwy = rw_mc_trajectories_2d(
                int(num_anim_paths),
                int(num_anim_steps),
                a,
                int(seed)
            )

            anim_rw_paths, fig_rw_paths = make_path_animation(
                rwx, rwy, "Random-Walk Trajectories"
            )
            rw_paths_gif = anim_to_base64_gif(anim_rw_paths)
            plt.close(fig_rw_paths)

            anim_rw_cloud, fig_rw_cloud = make_cloud_animation(
                rwx, rwy, "Random-Walk Particle Cloud"
            )
            rw_cloud_gif = anim_to_base64_gif(anim_rw_cloud)
            plt.close(fig_rw_cloud)

            lvx, lvy = levy_mc_trajectories_2d(
                int(num_anim_paths),
                int(num_anim_steps),
                mu,
                D1,
                dt_anim,
                int(seed) + 1000
            )

            anim_lv_paths, fig_lv_paths = make_path_animation(
                lvx, lvy, "Lévy-Flight Trajectories"
            )
            lv_paths_gif = anim_to_base64_gif(anim_lv_paths)
            plt.close(fig_lv_paths)

            anim_lv_cloud, fig_lv_cloud = make_cloud_animation(
                lvx, lvy, "Lévy-Flight Particle Cloud"
            )
            lv_cloud_gif = anim_to_base64_gif(anim_lv_cloud)
            plt.close(fig_lv_cloud)

        a1, a2 = st.columns(2)

        with a1:
            st.markdown("**Random-Walk Trajectories**")
            st.markdown(
                f'<img src="data:image/gif;base64,{rw_paths_gif}" width="100%">',
                unsafe_allow_html=True
            )
            st.caption(
                "Each walker takes steps of fixed length in a random direction. "
                "The result is a compact, gradually spreading cluster with no large jumps."
            )

            st.markdown("**Random-Walk Particle Cloud**")
            st.markdown(
                f'<img src="data:image/gif;base64,{rw_cloud_gif}" width="100%">',
                unsafe_allow_html=True
            )
            st.caption(
                "Because every step is the same length, the cloud grows smoothly and symmetrically "
                "outward, maintaining a roughly circular Gaussian shape at all times. The density "
                "falls off uniformly in every direction with no outliers."
            )

        with a2:
            st.markdown("**Lévy-Flight Trajectories**")
            st.markdown(
                f'<img src="data:image/gif;base64,{lv_paths_gif}" width="100%">',
                unsafe_allow_html=True
            )
            st.caption(
                "Step lengths are drawn from a heavy-tailed distribution, making occasional very long "
                "jumps far more likely than in a standard random walk. Notice how walkers can suddenly "
                "relocate far from their previous position — a signature of superdiffusion."
            )

            st.markdown("**Lévy-Flight Particle Cloud**")
            st.markdown(
                f'<img src="data:image/gif;base64,{lv_cloud_gif}" width="100%">',
                unsafe_allow_html=True
            )
            st.caption(
                "The cloud does not spread uniformly. Instead it retains a dense central core — "
                "where most walkers remain — while a small number of particles are scattered far "
                "from the origin by rare large jumps. This heavy-tailed shape persists at all times "
                "and is the visual signature of the power-law distribution."
            )

        # D_rw is needed for the Gaussian comparison curve
        _, _, D_rw, _ = rw_moments(N, a, Gamma, t_rw)

        # ----------------------------------------------------
        # Lévy section
        # ----------------------------------------------------
        x_grid = np.linspace(-xmax, xmax, int(nx))
        pdfs = [levy_pdf_fourier(x_grid, t, mu, D1) for t in times]
        widths, slope, intercept = levy_width_slope(times, x_grid, pdfs)

        # Monte Carlo comparison shown first, directly below animations
        if HAS_SCIPY:
            t_mc = times[len(times) // 2]
            p_mc = levy_pdf_fourier(x_grid, t_mc, mu, D1)
            p_gauss_mc = gaussian_pdf(x_grid, D_rw, t_mc)
            levy_samples = levy_mc_samples(mu, D1, t_mc, 30000, int(seed))

            st.subheader("Lévy vs Gaussian Distribution")
            st.pyplot(plot_levy_mc_vs_theory(x_grid, p_mc, p_gauss_mc, levy_samples, mu, t_mc))
            st.caption(
                "Both distributions are centered at zero, but their tails behave fundamentally "
                "differently. The Gaussian decays exponentially, meaning the probability of a large "
                "displacement drops off extremely fast. The Lévy distribution decays as a power law, "
                "which falls off far more slowly. "
                "No matter how wide the Gaussian is made, the Lévy tails will always extend further. "
                "Extreme events that are essentially impossible under normal diffusion remain genuinely "
                "probable in a Lévy flight."
            )
        else:
            st.info("SciPy not installed, so the optional Lévy Monte Carlo comparison was skipped.")

        d1, d2, d3 = st.columns(3)
        d1.metric("z", f"{mu:.4f}")
        d2.metric("d_c", f"{2 * mu - 2:.4f}")
        d3.metric("Width slope", f"{slope:.4f}")

        st.subheader("Lévy PDFs")
        st.pyplot(plot_levy_pdfs(x_grid, times, pdfs))
        st.caption(
            "Each curve shows the probability distribution of particle positions at a different "
            "point in time. As time progresses the distribution broadens and flattens, reflecting "
            "the spreading of the particle cloud. The heavy tails remain visible at all times — "
            "there is always a non-negligible probability of finding a particle far from the origin."
        )

        st.pyplot(plot_levy_width(times, widths, slope, intercept, mu))
        st.caption(
            f"The width of the distribution (measured by the interquartile range) is plotted against "
            f"time on a log-log scale. Theory predicts this width grows as t^(1/μ) — here "
            f"t^(1/{mu:.2f}) = t^({1/mu:.4f}). The estimated slope from the data is {slope:.4f}, "
            f"which should match the theoretical exponent of {1/mu:.4f}. A straight line on a "
            f"log-log plot confirms the power-law relationship."
        )

        st.markdown("---")
        st.markdown("""
## Conclusion

The standard random walk is an elegant model, but nature rarely moves in equal-sized steps. Real
systems are often governed by dynamics in which rare,
large events play an outsized role, and it is precisely these events that a Lévy flight captures.

The key insight is a simple one: replacing the Gaussian step-length distribution with a power-law
distribution changes the character of motion entirely. Spreading is faster, tails are always heavier,
and extreme displacements are never truly negligible. These are not just mathematical curiosities —
they show up in the real world.

When an albatross or a shark forages for food, GPS tracking reveals a movement pattern that closely
follows a Lévy flight: many short exploratory steps within a local area, punctuated by occasional
long-range relocations to a completely new region. This turns out to be close to the optimal
search strategy when food is sparse and unpredictably distributed, as the heavy-tailed jump lengths
allow the animal to escape depleted areas efficiently without wasting energy on purely random wandering.

In financial markets, the analogy is equally strong. Daily price returns are far better described
by a heavy-tailed distribution than a Gaussian one. The crashes and surges that appear as
statistical outliers under a normal distribution — once-in-a-century events by Gaussian logic —
occur at a much greater frequency in practice. Models that ignore this, as many
classical finance models do, systematically underestimate the probability of extreme moves.

The broader lesson is that when you observe a process driven by many independent random events,
the Gaussian is not always the right default. If the underlying step-length distribution has a
heavy tail, the collective behavior will too. The difference between the two, as this tool
shows, is both mathematically precise and visually dramatic.
""")

    except Exception as e:
        st.error(str(e))