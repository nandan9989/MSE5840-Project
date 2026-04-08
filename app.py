import os
import tempfile
import base64
from math import comb

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Needed so matplotlib works cleanly inside Streamlit
import matplotlib.pyplot as plt
import streamlit as st
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
st.write(
    "Interactive theory plots, Monte Carlo checks, and animations for "
    "standard random walks and Lévy flights."
)


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


def plot_levy_mc_vs_theory(x_grid, p_theory, samples, mu, t):
    """
    Compare Monte Carlo histogram against theoretical PDF.

    Improvements here:
    - more bins for smoother histogram
    - explicit x-limits for a cleaner comparison
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(samples, bins=300, density=True, alpha=0.35, label="Monte Carlo")
    ax.plot(x_grid, p_theory, linewidth=2.5, label="Theory")

    # Show a cleaner central region while still keeping visible tails
    ax.set_xlim(-50, 50)

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title(f"Lévy Monte Carlo vs Theory (μ={mu}, t={t})")
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

seed = st.sidebar.number_input("Random seed", value=42, step=1)

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
        # Random walk section
        # ----------------------------------------------------
        finals = rw_mc_final_positions(int(num_walkers), N, a, int(seed))
        mean_rw, msd_rw, D_rw, msd_diff = rw_moments(N, a, Gamma, t_rw)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("N", f"{N}")
        c2.metric("Theory <R²>", f"{msd_rw:.4f}")
        c3.metric("MC <R²>", f"{np.mean(finals**2):.4f}")
        c4.metric("2Dt", f"{msd_diff:.4f}")

        st.subheader("Random Walk")
        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(plot_rw_distribution(N, a, D_rw, t_rw, finals))

        with col2:
            Nvals = np.array([10, 20, 40, 60, 80, 100, min(N, 120)])
            Nvals = np.unique(Nvals[Nvals >= 1])
            st.pyplot(plot_rw_msd_check(Nvals, a, Gamma, int(num_walkers), int(seed)))

        # ----------------------------------------------------
        # Lévy section
        # ----------------------------------------------------
        x_grid = np.linspace(-xmax, xmax, int(nx))
        pdfs = [levy_pdf_fourier(x_grid, t, mu, D1) for t in times]
        widths, slope, intercept = levy_width_slope(times, x_grid, pdfs)

        d1, d2, d3 = st.columns(3)
        d1.metric("z", f"{mu:.4f}")
        d2.metric("d_c", f"{2 * mu - 2:.4f}")
        d3.metric("Width slope", f"{slope:.4f}")

        st.subheader("Lévy PDFs and Scaling")
        col3, col4 = st.columns(2)

        with col3:
            st.pyplot(plot_levy_pdfs(x_grid, times, pdfs))

        with col4:
            st.pyplot(plot_levy_scaling(x_grid, times, pdfs, mu))

        st.pyplot(plot_levy_width(times, widths, slope, intercept, mu))

        # Optional Monte Carlo comparison
        if HAS_SCIPY:
            t_mc = times[len(times) // 2]
            p_mc = levy_pdf_fourier(x_grid, t_mc, mu, D1)
            samples = levy_mc_samples(mu, D1, t_mc, 30000, int(seed))

            st.subheader("Lévy Monte Carlo vs Theory")
            st.pyplot(plot_levy_mc_vs_theory(x_grid, p_mc, samples, mu, t_mc))
        else:
            st.info("SciPy not installed, so the optional Lévy Monte Carlo comparison was skipped.")

        # ----------------------------------------------------
        # Animation section
        # ----------------------------------------------------
        st.subheader("Animations")

        with st.spinner("Generating animations..."):
            # Random-walk trajectory and cloud animations
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

            # Lévy trajectory and cloud animations
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

            st.markdown("**Random-Walk Particle Cloud**")
            st.markdown(
                f'<img src="data:image/gif;base64,{rw_cloud_gif}" width="100%">',
                unsafe_allow_html=True
            )

        with a2:
            st.markdown("**Lévy-Flight Trajectories**")
            st.markdown(
                f'<img src="data:image/gif;base64,{lv_paths_gif}" width="100%">',
                unsafe_allow_html=True
            )

            st.markdown("**Lévy-Flight Particle Cloud**")
            st.markdown(
                f'<img src="data:image/gif;base64,{lv_cloud_gif}" width="100%">',
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(str(e))