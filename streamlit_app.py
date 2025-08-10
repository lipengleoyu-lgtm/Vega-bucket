import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Asian Option – Bucket Vegas by Tenor", layout="wide")
st.title("Asian Option – Bucket Vegas by Tenor")

# -------------------- Sidebar Inputs --------------------
with st.sidebar:
    st.header("Option Parameters")
    S0 = st.number_input("Spot S0", value=100.0, min_value=0.0, step=1.0)
    K = st.number_input("Strike K", value=100.0, min_value=0.0, step=1.0)
    r = st.number_input("Risk-free rate r", value=0.03, step=0.001)
    q = st.number_input("Dividend yield q", value=0.00, step=0.001)
    T = st.number_input("Maturity (years)", value=1.0, min_value=0.1, step=0.1)

    st.header("Averaging Window")
    freq_choice = st.selectbox("Averaging frequency per year", ["Monthly", "Weekly", "Daily"], index=0)
    if freq_choice == "Monthly":
        buckets_per_year = 12
    elif freq_choice == "Weekly":
        buckets_per_year = 52
    else:
        buckets_per_year = 252
    N = max(2, int(round(buckets_per_year * T)))

    start_frac, end_frac = st.slider("Averaging window (fraction of maturity)", 0.0, 1.0, (0.0, 1.0), step=0.01)

    st.header("Volatility Settings")
    base_vol = st.number_input("Base vol (annualized)", value=0.20, min_value=0.0001, step=0.005)

    bump = st.number_input("Bump size (abs vol)", value=0.005, min_value=1e-5, step=0.001)
    greek_type = st.selectbox("Option type", ["Call", "Put"], index=0)

    st.header("Monte Carlo Controls")
    n_paths = st.number_input("Number of paths", value=50000, min_value=1000, step=5000)
    antithetic = st.checkbox("Use antithetic variates", value=True)
    seed = st.number_input("Random seed", value=42, step=1)

# -------------------- Functions --------------------

def simulate_price(sigmas, Z, S0, r, q, T, start_idx, end_idx, K, greek_type, antithetic):
    N = len(sigmas)
    dt = T / N
    mu = r - q
    if antithetic:
        Z = np.vstack([Z, -Z])
    log_S = np.full(Z.shape[0], np.log(S0))
    S_sum = np.zeros(Z.shape[0])
    count = max(1, end_idx - start_idx)
    for i in range(N):
        sigma_i = sigmas[i]
        log_S += (mu - 0.5 * sigma_i**2) * dt + sigma_i * np.sqrt(dt) * Z[:, i]
        if start_idx <= i < end_idx:
            S_sum += np.exp(log_S)
    A = S_sum / count
    payoff = np.maximum(A - K, 0.0) if greek_type == "Call" else np.maximum(K - A, 0.0)
    return np.exp(-r * T) * payoff.mean()

def compute_bucket_vegas(sigmas, Z, bump, *params):
    vegas = np.zeros(len(sigmas))
    for i in range(len(sigmas)):
        up = sigmas.copy(); up[i] += bump
        dn = sigmas.copy(); dn[i] -= bump
        p_up = simulate_price(up, Z, *params)
        p_dn = simulate_price(dn, Z, *params)
        vegas[i] = (p_up - p_dn) / (2 * bump)
    return vegas

# -------------------- Main Logic --------------------
sigmas = np.full(N, base_vol)
start_idx = int(np.floor(start_frac * N))
end_idx = int(np.ceil(end_frac * N))
start_idx = max(0, min(start_idx, N-1))
end_idx = max(start_idx+1, min(end_idx, N))

rng = np.random.default_rng(int(seed))
Z_base = rng.standard_normal(size=(int(n_paths), N))

with st.spinner("Running Monte Carlo..."):
     base_price = simulate_price(sigmas, Z_base, S0, r, q, T, start_idx, end_idx, K, greek_type, antithetic)
     vegas = compute_bucket_vegas(sigmas, Z_base, bump, S0, r, q, T, start_idx, end_idx, K, greek_type, antithetic)

vega_df = pd.DataFrame({"Bucket": np.arange(1, N+1), "Vega": vegas})
_total = vega_df["Vega"].sum()
vega_df["Normalized (sum=1)"] = vega_df["Vega"] / _total if _total != 0 else 0.0
vega_df["% of Total Vega"] = (100 * vega_df["Normalized (sum=1)"]).round(2)

# -------------------- Layout --------------------
left, right = st.columns([1.25, 1])
with left:
    st.subheader("Bucket Vegas by Tenor")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(vega_df["Bucket"], vega_df["Vega"])
    if freq_choice == "Monthly":
        ax.set_xlabel("Month index (1 = first month)")
    elif freq_choice == "Weekly":
        ax.set_xlabel("Week index (1 = first week)")
    else:
        ax.set_xlabel("Day index (1 = first day)")
    ax.set_ylabel("Bucket Vega (per abs vol)")
    ax.set_title("Asian Option Bucket Vegas")
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Summary")
    st.metric("Base Price", f"{base_price:.6f}")
    st.write("Buckets N:", N)
    st.write("Averaging buckets:", f"[{start_idx+1} .. {end_idx}] of {N}")
    st.dataframe(vega_df, use_container_width=True, height=420)

# -------------------- Downloads --------------------
st.download_button(
    "Download CSV",
    vega_df.to_csv(index=False).encode("utf-8"),
    file_name="asian_bucket_vegas.csv",
    mime="text/csv",
    key="download_csv_button",
)

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

