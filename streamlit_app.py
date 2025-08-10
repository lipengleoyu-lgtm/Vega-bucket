import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Asian Option – Bucket Vegas by Tenor", layout="wide")
st.title("Asian Option – Bucket Vegas by Tenor")
st.caption("Monte Carlo with piecewise-constant monthly/weekly/daily vol; bucket vega via common random numbers + central differences.")

# -------------------- Sidebar Inputs --------------------
with st.sidebar:
    st.header("Option Parameters")
    S0 = st.number_input("Spot S0", value=100.0, min_value=0.0, step=1.0, format="%0.6f")
    K = st.number_input("Strike K", value=100.0, min_value=0.0, step=1.0, format="%0.6f")
    r = st.number_input("Risk-free rate r", value=0.03, step=0.001, format="%0.6f")
    q = st.number_input("Dividend yield q", value=0.00, step=0.001, format="%0.6f")
    T = st.number_input("Maturity (years)", value=1.0, min_value=0.01, step=0.25, format="%0.6f")

    st.header("Averaging Window")
    freq = st.selectbox("Averaging frequency (number of buckets across full maturity)",
                        ["Monthly (12)", "Weekly (52)", "Daily (252)", "Custom"], index=0)
    if freq == "Monthly (12)":
        N = 12
    elif freq == "Weekly (52)":
        N = 52
    elif freq == "Daily (252)":
        N = 252
    else:
        N = st.number_input("Custom number of tenor buckets (>=2)", value=24, min_value=2, step=1)

    st.caption("The simulation uses N evenly spaced steps over [0, T]. Averaging can be over a subwindow.")
    start_frac, end_frac = st.slider("Averaging window (as fraction of T)", 0.0, 1.0, (0.0, 1.0), step=0.01)

    st.header("Volatility Term Structure")
    vol_mode = st.selectbox("Vol structure across buckets", ["Flat", "Linear slope", "Custom list"])
    base_vol = st.number_input("Base vol (annualized)", value=0.20, min_value=0.0001, step=0.005, format="%0.6f")

    if vol_mode == "Linear slope":
        end_vol = st.number_input("End vol (at last bucket)", value=0.20, min_value=0.0001, step=0.005, format="%0.6f")
    elif vol_mode == "Custom list":
        st.caption("Provide a comma-separated list of vols (length must equal the number of buckets N). Example: 0.2,0.205,0.21,...")
        custom_vol_text = st.text_area("Custom vols", value=", ".join([f"{base_vol:.4f}" for _ in range(N)]), height=100)

    st.header("Greeks Settings")
    bump = st.number_input("Bump size (absolute vol; e.g. 0.005 = 0.5 vol point)", value=0.005, min_value=1e-5, step=0.001, format="%0.6f")
    greek_type = st.selectbox("Option type", ["Call", "Put"], index=0)

    st.header("Monte Carlo Controls")
    n_paths = st.number_input("Number of paths", value=50000, min_value=1000, step=5000)
    antithetic = st.checkbox("Use antithetic variates", value=True)
    seed = st.number_input("Random seed", value=42, step=1)

# -------------------- Helper Functions --------------------

def parse_vol_term_structure(N, vol_mode, base_vol, end_vol=None, custom_text=None):
    if vol_mode == "Flat":
        sigmas = np.full(N, base_vol, dtype=float)
    elif vol_mode == "Linear slope":
        if end_vol is None:
            end_vol = base_vol
        sigmas = np.linspace(base_vol, end_vol, N, dtype=float)
    else:  # Custom list
        try:
            arr = [float(x) for x in custom_text.replace("\n", ",").split(",") if str(x).strip() != ""]
            sigmas = np.array(arr, dtype=float)
            if len(sigmas) != N:
                st.error(f"Custom vol list length {len(sigmas)} != N ({N}). Using flat vols instead.")
                sigmas = np.full(N, base_vol, dtype=float)
        except Exception:
            st.error("Failed to parse custom vol list. Using flat vols instead.")
            sigmas = np.full(N, base_vol, dtype=float)
    return sigmas


def simulate_price(sigmas, Z, S0, r, q, T, start_idx, end_idx, K, greek_type, antithetic):
    """
    Simulate GBM at N evenly-spaced buckets with piecewise-constant sigma per bucket.
    Uses exact monthly-like step per bucket i:
        S_i = S_{i-1} * exp((mu - 0.5*sigma_i^2)*dt + sigma_i*sqrt(dt)*Z_i)
    Returns discounted price of arithmetic-average option over indices [start_idx, end_idx) (inclusive start, exclusive end).
    """
    N = len(sigmas)
    dt = T / N
    mu = r - q

    if antithetic:
        Z = np.vstack([Z, -Z])

    n_eff = Z.shape[0]
    log_S = np.full(n_eff, np.log(S0), dtype=float)
    S_sum = np.zeros(n_eff, dtype=float)
    count = max(1, end_idx - start_idx)

    for i in range(N):
        sigma_i = sigmas[i]
        log_S += (mu - 0.5 * sigma_i**2) * dt + sigma_i * np.sqrt(dt) * Z[:, i]
        if start_idx <= i < end_idx:
            S_sum += np.exp(log_S)

    A = S_sum / count

    if greek_type == "Call":
        payoff = np.maximum(A - K, 0.0)
    else:
        payoff = np.maximum(K - A, 0.0)

    price = np.exp(-r * T) * payoff.mean()
    return price


def compute_bucket_vegas(sigmas, Z, bump, *params):
    N = len(sigmas)
    vegas = np.zeros(N, dtype=float)
    for i in range(N):
        up = sigmas.copy(); up[i] += bump
        dn = sigmas.copy(); dn[i] -= bump
        p_up = simulate_price(up, Z, *params)
        p_dn = simulate_price(dn, Z, *params)
        vegas[i] = (p_up - p_dn) / (2.0 * bump)
    return vegas

# -------------------- Build Inputs --------------------
if vol_mode == "Linear slope":
    sigmas = parse_vol_term_structure(N, vol_mode, base_vol, end_vol=end_vol)
elif vol_mode == "Custom list":
    sigmas = parse_vol_term_structure(N, vol_mode, base_vol, custom_text=custom_vol_text)
else:
    sigmas = parse_vol_term_structure(N, vol_mode, base_vol)

# Averaging window indices
start_idx = int(np.floor(start_frac * N))
end_idx = int(np.ceil(end_frac * N))
start_idx = max(0, min(start_idx, N-1))
end_idx = max(start_idx+1, min(end_idx, N))

# Common random numbers (CRN)
rng = np.random.default_rng(int(seed))
Z_base = rng.standard_normal(size=(int(n_paths), N))

# -------------------- Compute --------------------
with st.spinner("Running Monte Carlo..."):
    # Base price (not strictly needed for vegas but shown for context)
    base_price = simulate_price(sigmas, Z_base, S0, r, q, T, start_idx, end_idx, K, greek_type, antithetic)
    vegas = compute_bucket_vegas(sigmas, Z_base, bump, S0, r, q, T, start_idx, end_idx, K, greek_type, antithetic)

# Normalize and mask: show zero outside averaging window for clarity? We'll keep full buckets because vol bumps propagate.
vega_df = pd.DataFrame({
    "Bucket": np.arange(1, N+1),
    "Vega": vegas,
})
vega_df["Normalized (sum=1)"] = vega_df["Vega"] / vega_df["Vega"].sum() if vega_df["Vega"].sum() != 0 else 0.0
vega_df["% of Total Vega"] = (100 * vega_df["Normalized (sum=1)"]).round(2)

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Bucket Vegas by Tenor")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(vega_df["Bucket"], vega_df["Vega"])  # Do not set colors/styles per instruction simplicity
    ax.set_xlabel("Bucket (1 = earliest)")
    ax.set_ylabel("Bucket Vega (per abs vol)")
    ax.set_title("Asian Option Bucket Vegas")
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Summary")
    st.metric("Base price", f"{base_price:0.6f}")
    st.write("Averaging over buckets:", f"[{start_idx+1} .. {end_idx}] of {N}")
    st.dataframe(vega_df, use_container_width=True, height=420)

# -------------------- Downloads --------------------
csv = vega_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, file_name="asian_bucket_vegas.csv", mime="text/csv")

# -------------------- Notes --------------------
st.markdown(
    """
**Notes**
- Bucket vega is computed with **central finite differences** of size `bump` using **common random numbers** for variance reduction.
- Volatility is **piecewise-constant per bucket**. A bump in bucket *i* affects the distribution of all **later** buckets (GBM propagation), so shapes are typically **front-loaded** within the averaging window.
- Increase path count for smoother curves. Antithetic variates help.
- Set the averaging window to, e.g., `[0.5, 1.0]` to study bucket vegas for fixings in the **second half** of the maturity.
"""
)
