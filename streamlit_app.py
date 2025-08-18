import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Asian Option – Bucket Vegas & Hedge Optimizer", layout="wide")
st.title("Asian Option – Bucket Vegas & Hedge Optimizer")


page = st.sidebar.radio("Select Tool", ["Bucket Vega by Tenor", "Vega Hedge Maturity"])

# ==================== Utilities shared across tabs ====================

def exp_distance_function(t1, t2):
    return np.exp(abs(t1 - t2))

def linear_plus_one_distance(t1, t2):
    return abs(t1 - t2) + 1.0

DIST_FUNCS = {
    "|t1 - t2| + 1": linear_plus_one_distance,
    "exp(|t1 - t2|)": exp_distance_function
}

MAT_MAP = {
    "1m": 1/12,
    "3m": 3/12,
    "6m": 6/12,
    "1y": 1.0,
    "2y": 2.0,
}

# ==================== Tabs ====================
#TAB_BUCKETS, TAB_OPT = st.tabs(["Bucket Vegas", "Hedge Maturity Optimizer"]) 

# -------------------- Page 1: Bucket Vega by Tenor --------------------
if page == "Bucket Vega by Tenor":
    st.title("Asian Option – Bucket Vegas by Tenor")
    S0 = st.number_input("Spot S0", value=100.0, min_value=0.0, step=1.0)
    K = st.number_input("Strike K", value=100.0, min_value=0.0, step=1.0)
    r = st.number_input("Risk-free rate r", value=0.03, step=0.001)
    q = st.number_input("Dividend yield q", value=0.00, step=0.001)
    T = st.number_input("Maturity (years)", value=1.0, min_value=0.1, step=0.1)
    freq_choice = st.selectbox("Averaging frequency per year", ["Monthly", "Weekly", "Daily"], index=0)
    if freq_choice == "Monthly":
        buckets_per_year = 12
    elif freq_choice == "Weekly":
        buckets_per_year = 52
    else:
        buckets_per_year = 252
    N = max(2, int(round(buckets_per_year * T)))
    start_frac, end_frac = st.slider("Averaging window (fraction of maturity)", 0.0, 1.0, (0.0, 1.0), step=0.01)
    base_vol = st.number_input("Base vol (annualized)", value=0.20, min_value=0.0001, step=0.005)
    bump = st.number_input("Bump size (abs vol)", value=0.005, min_value=1e-5, step=0.001)
    bump_mode = st.selectbox(
        "Bump definition",
        ["Forward-bucket (OAT)", "Expiry-slice (increase expiry vol at t_j)"],
        index=0,
        key="bump_mode"
    )
    greek_type = st.selectbox("Option type", ["Call", "Put"], index=0)
    n_paths = st.number_input("Number of paths", value=50000, min_value=1000, step=5000)
    antithetic = st.checkbox("Use antithetic variates", value=True)
    seed = st.number_input("Random seed", value=42, step=1)

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

    def _price_from_forwards(S0, K, r, q, sigmas_fwd, Z, T, which, fixing_idx):
        start_idx, end_idx = fixing_idx
        return simulate_price(np.array(sigmas_fwd, dtype=float), Z, S0, r, q, T, start_idx, end_idx, K, "Call" if which=="call" else "Put", antithetic)

    def expiry_slice_vegas(S0, K, r, q, sigmas_fwd, Z, bump_abs=0.01, T=1.0, which="call", fixing_idx=None):
        """Bump the *expiry* vol at t_j by bump_abs by adjusting ONLY bucket j to increase sqrt(Var(0->t_j)/t_j) by bump_abs, keeping earlier buckets unchanged."""
        n = len(sigmas_fwd)
        dt = T / n
        base = _price_from_forwards(S0, K, r, q, sigmas_fwd, Z, T, which, fixing_idx)
        vegas = []
        var_cum = np.cumsum((np.array(sigmas_fwd)**2) * dt)
        for j in range(n):
            t_j = (j+1) * dt
            var_j = var_cum[j]
            sigma_exp_j = np.sqrt(max(var_j, 0.0) / t_j)
            target_var_j = (sigma_exp_j + bump_abs)**2 * t_j
            dVar = target_var_j - var_j
            var_bucket_j = (sigmas_fwd[j]**2) * dt + dVar
            if var_bucket_j < 0:
                var_bucket_j = 0.0
            bumped = np.array(sigmas_fwd, dtype=float)
            bumped[j] = np.sqrt(var_bucket_j / dt)
            p = _price_from_forwards(S0, K, r, q, bumped, Z, T, which, fixing_idx)
            vegas.append((p - base) / bump_abs)
        return np.array(vegas), base

    sigmas = np.full(N, base_vol)
    start_idx = int(np.floor(start_frac * N))
    end_idx   = int(np.ceil(end_frac  * N))
    start_idx = max(0, min(start_idx, N-1))
    end_idx   = max(start_idx + 1, min(end_idx, N))
    rng = np.random.default_rng(int(seed))
    Z_base = rng.standard_normal(size=(int(n_paths), N))
    # Compute base price and vegas depending on bump mode
    if bump_mode.startswith("Expiry-slice"):
        which = "call" if greek_type == "Call" else "put"
        vegas, base_price = expiry_slice_vegas(
            S0, K, r, q, sigmas, Z_base,
            bump_abs=bump, T=T, which=which,
            fixing_idx=(start_idx, end_idx)
        )
    else:
        base_price = simulate_price(sigmas, Z_base, S0, r, q, T, start_idx, end_idx, K, greek_type, antithetic)
        vegas = compute_bucket_vegas(sigmas, Z_base, bump, S0, r, q, T, start_idx, end_idx, K, greek_type, antithetic)

    vega_df = pd.DataFrame({"Bucket": np.arange(1, N+1), "Vega": vegas})
    _total = vega_df["Vega"].sum()
    vega_df["Normalized (sum=1)"] = vega_df["Vega"] / _total if _total != 0 else 0.0
    vega_df["% of Total Vega"] = (100 * vega_df["Normalized (sum=1)"]).round(2)

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

    st.download_button(
        "Download CSV",
        vega_df.to_csv(index=False).encode("utf-8"),
        file_name="asian_bucket_vegas.csv",
        mime="text/csv",
        key="download_csv_button",
    )

    st.markdown("**Notes**: Bump modes — Forward-bucket (OAT) bumps one forward bucket; Expiry-slice bumps expiry vol at t_j by adjusting only bucket j. Uses CRN; shapes often front-loaded under GBM.")


# ==================== TAB 2: Hedge Maturity Optimizer ====================
if page == "Vega Hedge Maturity":
    st.title("Optimal Vega Hedge Maturity")
    st.markdown("Provide per-ticker vega exposures at standard maturities (1m, 3m, 6m, 1y, 2y). The optimizer returns the maturity (or maturities) that minimize the distance-weighted cost:")
    st.latex(r"C(t_1) = \sum_{t_2} \mathrm{vega}(t_2)\, d(t_1, t_2)")

    # Distance function choice
    dist_choice = st.selectbox("Distance function", list(DIST_FUNCS.keys()))
    dist_func = DIST_FUNCS[dist_choice]

    # Editable table with dynamic rows
    sample = pd.DataFrame([
        {"Ticker": "SPX", "1m": 100, "3m": 50, "6m": 10, "1y": 5,  "2y": 100},
        {"Ticker": "AAPL", "1m": 20, "3m": 20, "6m": 10, "1y": 10, "2y": 0},
        {"Ticker": "MSFT", "1m": 10, "3m": 50, "6m": 10, "1y": 5,  "2y": 0},
        {"Ticker": "GOOG", "1m": 10, "3m": 50, "6m": 100, "1y": 50,  "2y": 10},
        {"Ticker": "NVDA", "1m": 10, "3m": 10, "6m": 10, "1y": 50,  "2y": 100},
        {"Ticker": "META", "1m": 20, "3m": 50, "6m": 20, "1y": 100,  "2y": 10},
    ])

    st.caption("Edit the exposures (vega in your units). Add/remove rows as needed.")
    df = st.data_editor(
        sample,
        num_rows="dynamic",
        use_container_width=True,
        key="hedge_table",
    )

    # Clean & compute per-ticker optimal maturities
    def row_to_dict(row: pd.Series) -> dict:
        d = {}
        for col in ["1m", "3m", "6m", "1y", "2y"]:
            val = row.get(col, 0)
            if pd.isna(val):
                continue
            try:
                v = float(val)
            except Exception:
                continue
            if v != 0.0:
                d[MAT_MAP[col]] = v
        return d

    def find_optimize_maturity(vega_exposure: dict, distance_function) -> list:
        maturities = list(vega_exposure.keys())
        if len(maturities) == 0:
            return []
        cost_dict = {}
        for t1 in maturities:
            cost = 0.0
            for t2 in maturities:
                cost += vega_exposure[t2] * distance_function(t1, t2)
            cost_dict[t1] = cost
        if len(cost_dict) == 0:
            return []
        min_cost = min(cost_dict.values())
        mins = [k for k, v in cost_dict.items() if v == min_cost]
        return mins

    results = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip() or "(untitled)"
        exp_dict = row_to_dict(row)
        mins = find_optimize_maturity(exp_dict, dist_func)
        # present results as friendly labels (e.g., '3m', '1y')
        label_map = {v: k for k, v in MAT_MAP.items()}
        labels = [label_map.get(t, f"{t:g}y") for t in mins]
        labels = [labels[-1]]
        results.append({
            "Ticker": ticker,
            "Optimal hedge maturity/maturities": ", ".join(labels) if labels else "—",
        })

    res_df = pd.DataFrame(results)

    left2, right2 = st.columns([1, 1])
    with left2:
        st.subheader("Optimizer Results")
        st.dataframe(res_df, use_container_width=True)

    with right2:
        st.subheader("Details (current row)")
        if len(df) > 0:
            r0 = df.iloc[-1]
            st.write("Example exposure (first row):", row_to_dict(r0))
        st.markdown(
            f"**Distance:** `{dist_choice}`  ")

    # Download
    st.download_button(
        "Download results CSV",
        res_df.to_csv(index=False).encode("utf-8"),
        file_name="hedge_optimizer_results.csv",
        mime="text/csv",
        key="download_results_btn",
    )

    # Notes
    st.markdown(
        """
**Notes**
- Optimizer searches over the *existing exposure maturities* only (discrete set). If you want to hedge at maturities not in the table, add a small exposure column for that tenor to include it as a candidate.
- You can change the distance function to reflect your desk’s preference for near-tenor vs. far-tenor hedges.
- All vega numbers are taken as-is (no sign conventions enforced); use positive for long-vega, negative for short-vega exposures as appropriate.
"""
    )

