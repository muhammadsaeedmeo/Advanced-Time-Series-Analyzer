# ======================================================================
# 🟩 ADVANCED TIME SERIES ANALYZER – Single-file Streamlit App (app.py)
# ======================================================================
# Fixed version with improved econometric rigor
# Run locally: `streamlit run app.py`
# ======================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ======================================================================
# 🟩 PASSWORD PROTECTION
# ======================================================================
def check_password():
    """Returns `True` if the user entered the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "1992":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Enter Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please enter the password to access the application*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Enter Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# ======================================================================
# 🟩 BASIC CONFIGURATION
# ======================================================================
st.set_page_config(layout="wide", page_title="Advanced Time Series Analyzer")
st.title("📊 Advanced Time Series Analysis & Visualization")

# ======================================================================
# 🟩 SECTION 1: DATA INPUT & PREPARATION
# ======================================================================
st.sidebar.header("1. Data Input")
use_sample = st.sidebar.checkbox("Use sample dataset (provided)", value=True)
uploaded_file = None
if not use_sample:
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if use_sample:
    try:
        df = pd.read_csv("sample_time_series.csv", parse_dates=["date"])
    except Exception:
        dates = pd.date_range(start="2023-01-01", periods=365)
        df = pd.DataFrame({
            "date": dates,
            "series_a": np.random.randn(len(dates)).cumsum() + 10,
            "series_b": np.random.randn(len(dates)).cumsum() + 20
        })
else:
    if uploaded_file is None:
        st.info("Please upload a CSV/Excel file or enable 'Use sample dataset' in the sidebar.")
        st.stop()
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()

# 🟩 DATETIME COLUMN DETECTION
st.sidebar.header("2. Datetime Settings")
possible_dt = [c for c in df.columns if "date" in c.lower() or np.issubdtype(df[c].dtype, np.datetime64)]
if possible_dt:
    dt_col = st.sidebar.selectbox("Select datetime column", options=possible_dt, index=0)
else:
    dt_col = st.sidebar.selectbox("Select datetime column", options=df.columns)
    try:
        df[dt_col] = pd.to_datetime(df[dt_col])
    except Exception:
        st.error("Could not parse the chosen datetime column. Ensure it contains ISO-like dates.")
        st.stop()

df[dt_col] = pd.to_datetime(df[dt_col])
df = df.sort_values(dt_col).reset_index(drop=True)
df.set_index(dt_col, inplace=True)

# ======================================================================
# 🟩 SECTION 2: UI CUSTOMIZATION & FILTERS
# ======================================================================
st.sidebar.header("3. Visualization & Filters")
plot_backend = st.sidebar.selectbox("Plot engine", ["Plotly (interactive)", "Matplotlib (static)"])
theme = st.sidebar.selectbox("Theme/template", ["default", "plotly_white", "plotly_dark", "seaborn", "classic"])
bg_color = st.sidebar.color_picker("Background color", "#ffffff")
show_grid = st.sidebar.checkbox("Show grid lines", value=True)
line_width = st.sidebar.slider("Line width", min_value=1, max_value=6, value=2)
marker_size = st.sidebar.slider("Marker size (for plotly)", min_value=4, max_value=12, value=6)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns detected. Upload a dataset with numeric time series columns.")
    st.stop()

# 🟩 Numeric filter (optional)
st.sidebar.header("4. Numeric Filter (optional)")
filter_col = st.sidebar.selectbox("Filter rows by numeric column", options=[None] + numeric_cols, index=0)
if filter_col:
    minv, maxv = float(df[filter_col].min()), float(df[filter_col].max())
    lo, hi = st.sidebar.slider("Filter range", min_value=minv, max_value=maxv, value=(minv, maxv))
    df = df[(df[filter_col] >= lo) & (df[filter_col] <= hi)]

# ======================================================================
# 🟩 SECTION 3: DESCRIPTIVE STATISTICS & DATA OVERVIEW
# ======================================================================
import io

st.header("🧾 Descriptive Statistics & Data Overview")

data_option = st.radio(
    "Select data type for descriptive statistics:",
    ["Raw Data", "Log-Transformed Data"],
    horizontal=True
)

if data_option == "Log-Transformed Data":
    log_df = df.copy()
    for col in log_df.select_dtypes(include=[np.number]).columns:
        log_df[col] = log_df[col].apply(lambda x: np.log(x) if x > 0 else np.nan)
    display_df = log_df
else:
    display_df = df

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{data_option} Preview")
    st.dataframe(display_df.head(50), use_container_width=True)

with col2:
    st.subheader(f"{data_option} Summary Statistics")
    summary_df = display_df.describe(include="all").round(3)
    st.dataframe(summary_df, use_container_width=True)

excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
    summary_df.to_excel(writer, index=True, sheet_name=f"{data_option}_Stats")

st.download_button(
    label=f"📥 Download {data_option} Summary Statistics (Excel)",
    data=excel_buf.getvalue(),
    file_name=f"{data_option.lower().replace(' ', '_')}_summary_statistics.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ======================================================================
# 🟩 SECTION 4: VARIABLE SELECTION
# ======================================================================
st.header("🔎 Variable Selection")
dep_var = st.selectbox("Dependent variable (Y)", options=numeric_cols, index=0)
indep_var = st.selectbox("Independent variable (X) — optional", options=[None] + numeric_cols, index=0)

# ======================================================================
# 🟩 SECTION 5: TIME SERIES PLOT
# ======================================================================
st.header("📈 Time Series Plot")

plot_mode = st.radio(
    "Select Plot Mode:",
    options=["Single Variable", "All Variables"],
    index=0,
    horizontal=True
)

if plot_backend.startswith("Plotly"):
    fig = go.Figure()

    if plot_mode == "Single Variable":
        fig.add_trace(go.Scatter(
            x=df.index, y=df[dep_var], mode="lines+markers",
            name=dep_var, line=dict(width=line_width), marker=dict(size=marker_size)
        ))
        if indep_var and indep_var != dep_var:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[indep_var], mode="lines",
                name=indep_var, line=dict(width=line_width)
            ))
    else:
        for col in numeric_cols:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode="lines",
                name=col, line=dict(width=line_width)
            ))

    fig.update_layout(
        title="Time Series Plot" if plot_mode == "All Variables" else f"Time Series: {dep_var}",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        legend_title="Variables"
    )
    if not show_grid:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    if theme == "seaborn":
        sns.set()
    else:
        plt.style.use('classic' if theme == "classic" else 'default')

    fig, ax = plt.subplots(figsize=(12, 4))

    if plot_mode == "Single Variable":
        ax.plot(df.index, df[dep_var], linewidth=line_width, marker='o', markersize=marker_size/2, label=dep_var)
        if indep_var and indep_var != dep_var:
            ax.plot(df.index, df[indep_var], linewidth=line_width, alpha=0.8, label=indep_var)
    else:
        for col in numeric_cols:
            ax.plot(df.index, df[col], linewidth=line_width, label=col)

    ax.set_facecolor(bg_color)
    ax.grid(show_grid)
    ax.set_title("Time Series Plot" if plot_mode == "All Variables" else f"Time Series: {dep_var}")
    ax.legend(loc="upper right", fontsize="small")
    st.pyplot(fig)

# ======================================================================
# 🟩 SECTION 6: ACF & PACF PLOTS (NEW - ESSENTIAL FOR ECONOMETRICS)
# ======================================================================
st.header("📊 Autocorrelation Analysis (ACF & PACF)")

st.markdown("""
**Purpose:** ACF and PACF plots help identify the order of AR(p) and MA(q) components for ARIMA modeling.
- **ACF**: Shows correlation between observations at different lags
- **PACF**: Shows partial correlation after removing effects of shorter lags
""")

acf_var = st.selectbox("Select variable for ACF/PACF analysis", numeric_cols, key="acf_var")
max_lags_acf = st.slider("Maximum lags to display", min_value=10, max_value=100, value=40, key="acf_lags")

col_acf1, col_acf2 = st.columns(2)

with col_acf1:
    fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
    plot_acf(df[acf_var].dropna(), lags=max_lags_acf, ax=ax_acf, alpha=0.05)
    ax_acf.set_title(f"Autocorrelation Function (ACF) - {acf_var}")
    st.pyplot(fig_acf)

with col_acf2:
    fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
    plot_pacf(df[acf_var].dropna(), lags=max_lags_acf, ax=ax_pacf, alpha=0.05, method='ywm')
    ax_pacf.set_title(f"Partial Autocorrelation Function (PACF) - {acf_var}")
    st.pyplot(fig_pacf)

# Ljung-Box Test for autocorrelation
st.subheader("Ljung-Box Test for Serial Correlation")
st.markdown("**H0:** No serial correlation up to lag k | **H1:** Serial correlation exists")

lb_lags = st.slider("Lags for Ljung-Box test", min_value=1, max_value=50, value=10, key="lb_lags")
lb_result = acorr_ljungbox(df[acf_var].dropna(), lags=lb_lags, return_df=True)
lb_result['Significant'] = lb_result['lb_pvalue'] < 0.05
st.dataframe(lb_result.round(4), use_container_width=True)

if lb_result['lb_pvalue'].iloc[-1] < 0.05:
    st.warning("⚠️ Serial correlation detected. Consider using ARIMA/ARMA models.")
else:
    st.success("✅ No significant serial correlation detected.")

# ======================================================================
# 🟩 SECTION 7: SCATTER PLOT
# ======================================================================
st.header("🔹 Scatter Plot")
sx = st.selectbox("Scatter X (independent)", options=numeric_cols, index=0, key="sx")
sy = st.selectbox("Scatter Y (dependent)", options=numeric_cols, index=min(1, len(numeric_cols)-1), key="sy")
color_by_options = [None] + [c for c in df.columns if df[c].nunique() < 50]
color_by = st.selectbox("Color by (categorical) — optional", options=color_by_options, index=0)

if plot_backend.startswith("Plotly"):
    scdf = df.reset_index()
    fig_s = px.scatter(scdf, x=sx, y=sy, color=color_by if color_by else None,
                       title=f"{sy} vs {sx}", trendline="ols")
    fig_s.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color)
    st.plotly_chart(fig_s, use_container_width=True)
else:
    fig_s, axs = plt.subplots(figsize=(8, 5))
    axs.scatter(df[sx], df[sy], s=20, alpha=0.6)
    z = np.polyfit(df[sx].dropna(), df[sy].dropna(), 1)
    p = np.poly1d(z)
    axs.plot(df[sx], p(df[sx]), "r--", alpha=0.8, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    axs.set_facecolor(bg_color)
    axs.grid(show_grid)
    axs.set_title(f"{sy} vs {sx}")
    axs.set_xlabel(sx)
    axs.set_ylabel(sy)
    axs.legend()
    st.pyplot(fig_s)

# ======================================================================
# 🟩 SECTION 8: CORRELATION HEATMAP
# ======================================================================
st.header("📉 Correlation Heatmap")

st.sidebar.header("Heatmap Settings")
heatmap_palette = st.sidebar.selectbox(
    "Select heatmap color palette",
    ["coolwarm", "viridis", "plasma", "cividis", "magma", "crest", "rocket", "Spectral", "icefire", "vlag"],
    index=0
)

corr = df[numeric_cols].corr()
fig_c, axc = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap=heatmap_palette,
    ax=axc,
    cbar_kws={"shrink": 0.8}
)
axc.set_facecolor(bg_color)
axc.set_title("Correlation Heatmap", fontsize=14, weight="bold")
st.pyplot(fig_c)

# ======================================================================
# 🟩 SECTION 9: DISTRIBUTION COMPARISON & NORMALITY TESTS
# ======================================================================
st.header("📊 Distribution Comparison & Normality Assessment")

col_to_plot = st.selectbox("Select numeric variable for distribution analysis", numeric_cols, key="dist_col")

st.sidebar.header("Distribution Chart Settings")
sns_style = st.sidebar.selectbox("Select Seaborn Style", ["whitegrid", "darkgrid", "white", "ticks", "dark"], index=1)
plt_style = st.sidebar.selectbox("Select Plot Style", ["default", "seaborn-v0_8-colorblind", "seaborn-v0_8-poster", "classic"], index=0)
sns.set_style(sns_style)
plt.style.use(plt_style)

def create_distribution_chart(data_series, y_label, title):
    """Creates comprehensive distribution visualization with 4 plot types"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    
    mean_val = data_series.mean()
    std_val = data_series.std()
    sns.barplot(x=["Mean"], y=[mean_val], ax=axes[0], color="skyblue")
    axes[0].errorbar(x=[0], y=[mean_val], yerr=std_val, fmt='o', color='black', capsize=5)
    axes[0].set_xlabel("Mean & Std Dev")
    axes[0].set_ylabel(y_label)
    axes[0].set_title("Bar Plot")

    sns.boxplot(y=data_series, ax=axes[1], color="lightgreen")
    axes[1].set_xlabel("Quartile Box")
    axes[1].set_title("Box Plot")

    sns.violinplot(y=data_series, ax=axes[2], color="lightcoral")
    axes[2].set_xlabel("Density")
    axes[2].set_title("Violin Plot")

    sns.stripplot(y=data_series, ax=axes[3], color="gray", jitter=True, alpha=0.6)
    axes[3].axhline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.2f}")
    axes[3].set_xlabel("Data Points")
    axes[3].set_title("Strip Plot")
    axes[3].legend(loc="upper right", fontsize="small")

    fig.suptitle(title, fontsize=16, weight="bold")
    fig.tight_layout(pad=2)
    return fig

if col_to_plot:
    fig = create_distribution_chart(df[col_to_plot].dropna(), y_label=col_to_plot, title=f"Distribution Overview: {col_to_plot}")
    st.pyplot(fig)
    
    st.markdown("""
    **📖 How to Interpret:**
    - **Bar Plot:** Shows mean value with error bars (±1 standard deviation)
    - **Box Plot:** Displays quartiles (Q1, median, Q3) and outliers
    - **Violin Plot:** Combines box plot with kernel density estimation
    - **Strip Plot:** Shows all individual data points with mean reference line
    """)
    
    # Normality test
    st.subheader("🔬 Normality Tests")
    data_clean = df[col_to_plot].dropna()
    
    # Jarque-Bera test
    jb_stat, jb_pval = stats.jarque_bera(data_clean)
    
    # Shapiro-Wilk test (use sample if too large)
    if len(data_clean) > 5000:
        sw_stat, sw_pval = stats.shapiro(data_clean.sample(5000, random_state=42))
        st.info("Note: Shapiro-Wilk test performed on random sample of 5000 observations")
    else:
        sw_stat, sw_pval = stats.shapiro(data_clean)
    
    norm_results = pd.DataFrame({
        'Test': ['Jarque-Bera', 'Shapiro-Wilk'],
        'Statistic': [round(jb_stat, 4), round(sw_stat, 4)],
        'p-value': [round(jb_pval, 4), round(sw_pval, 4)],
        'Normal?': ['Yes ✓' if jb_pval > 0.05 else 'No ✗', 'Yes ✓' if sw_pval > 0.05 else 'No ✗']
    })
    
    st.dataframe(norm_results, use_container_width=True)
    
    st.markdown("""
    **📖 Interpretation Guide:**
    - **H₀ (Null Hypothesis):** Data follows a normal distribution
    - **H₁ (Alternative Hypothesis):** Data does NOT follow a normal distribution
    - **Decision Rule:** If p-value < 0.05, reject H₀ (data is not normal)
    - **Jarque-Bera Test:** Based on skewness and kurtosis
    - **Shapiro-Wilk Test:** More powerful for small to medium sample sizes
    """)

# ======================================================================
# 🟩 SECTION 10: SEASONAL DECOMPOSITION (IMPROVED)
# ======================================================================
st.header("🔬 Seasonal Decomposition & Residuals")

decomp_method = st.radio("Decomposition method", ["Classical (Additive)", "Classical (Multiplicative)", "STL (Robust)"], horizontal=True)
period = st.number_input("Seasonal period (observations)", min_value=2, max_value=2000, value=12)

try:
    if decomp_method == "STL (Robust)":
        stl = STL(df[dep_var].dropna(), period=int(period), robust=True)
        decomp = stl.fit()
    else:
        model_type = 'additive' if 'Additive' in decomp_method else 'multiplicative'
        decomp = seasonal_decompose(df[dep_var].dropna(), period=int(period), model=model_type, extrapolate_trend='freq')
    
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=decomp.observed.index, y=decomp.observed, name="Observed"))
    fig_d.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, name="Trend"))
    fig_d.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, name="Seasonal"))
    fig_d.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, name="Residual"))
    fig_d.update_layout(title=f"Decomposition: {dep_var} ({decomp_method})", plot_bgcolor=bg_color, paper_bgcolor=bg_color)
    st.plotly_chart(fig_d, use_container_width=True)
    
    # Residual diagnostics
    st.subheader("Residual Diagnostics")
    residuals = decomp.resid.dropna()
    
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Mean", f"{residuals.mean():.4f}")
        st.metric("Std Dev", f"{residuals.std():.4f}")
    with col_r2:
        st.metric("Min", f"{residuals.min():.4f}")
        st.metric("Max", f"{residuals.max():.4f}")
    with col_r3:
        st.metric("Skewness", f"{stats.skew(residuals):.4f}")
        st.metric("Kurtosis", f"{stats.kurtosis(residuals):.4f}")
    
    st.markdown("**Ideal residuals:** Mean ≈ 0, no autocorrelation, normally distributed")
    
except Exception as e:
    st.warning(f"Decomposition failed: {e}")

# ======================================================================
# 🟩 SECTION 11: GRANGER CAUSALITY TEST (FIXED - ALL LAGS)
# ======================================================================
from statsmodels.tsa.stattools import grangercausalitytests

st.header("🔁 Granger Causality Test")

st.markdown("""
**Purpose:** Tests whether past values of X help predict Y (beyond what Y's own past values provide).
- **H0:** X does NOT Granger-cause Y
- **H1:** X DOES Granger-cause Y
- **Important:** Both series should be stationary (I(0)) for valid results
""")

dependent_var = st.selectbox("Select Dependent Variable (Y)", df.columns, key="gc_dep")
independent_vars = st.multiselect("Select Independent Variable(s) (X)", [c for c in df.columns if c != dependent_var])
max_lag = st.selectbox("Maximum Lag Order to test", list(range(1, 21)), index=4)

if st.button("Run Granger Causality Test"):
    try:
        results_list = []

        for indep in independent_vars:
            temp_df = df[[dependent_var, indep]].dropna()
            
            if len(temp_df) < max_lag + 10:
                st.warning(f"⚠️ Insufficient observations for {indep} ➜ {dependent_var}")
                continue

            # Test ALL lags from 1 to max_lag
            gc_res = grangercausalitytests(temp_df, maxlag=max_lag, verbose=False)
            
            # Forward direction: X ➜ Y
            for lag in range(1, max_lag + 1):
                f_test = gc_res[lag][0]['ssr_ftest'][0]
                p_value = gc_res[lag][0]['ssr_ftest'][1]
                results_list.append({
                    "Direction": f"{indep} ➜ {dependent_var}",
                    "Lag": lag,
                    "F-statistic": round(f_test, 4),
                    "p-value": round(p_value, 4),
                    "Significant?": "Yes ✓" if p_value < 0.05 else "No"
                })

            # Reverse direction: Y ➜ X
            temp_df_rev = df[[indep, dependent_var]].dropna()
            gc_rev = grangercausalitytests(temp_df_rev, maxlag=max_lag, verbose=False)
            
            for lag in range(1, max_lag + 1):
                f_test_rev = gc_rev[lag][0]['ssr_ftest'][0]
                p_value_rev = gc_rev[lag][0]['ssr_ftest'][1]
                results_list.append({
                    "Direction": f"{dependent_var} ➜ {indep}",
                    "Lag": lag,
                    "F-statistic": round(f_test_rev, 4),
                    "p-value": round(p_value_rev, 4),
                    "Significant?": "Yes ✓" if p_value_rev < 0.05 else "No"
                })

        gc_df = pd.DataFrame(results_list)

        st.subheader("📊 Granger Causality Results (All Lags)")
        st.dataframe(gc_df, use_container_width=True)
        
        st.markdown("""
        **Interpretation Guide:**
        - **p-value < 0.05**: Reject H0 → Granger causality exists at that lag
        - **p-value ≥ 0.05**: Fail to reject H0 → No Granger causality detected
        - Look for **multiple significant lags** for stronger evidence
        """)

        st.markdown("#### 📋 Copy Results")
        st.code(gc_df.to_markdown(index=False), language="markdown")

        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            gc_df.to_excel(writer, index=False, sheet_name="GrangerResults")

        st.download_button(
            label="📥 Download Results (Excel)",
            data=excel_buf.getvalue(),
            file_name="granger_causality_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Error while running Granger Causality Test: {e}")

# ======================================================================
# 🟩 SECTION: PROPER QUANTILE-ON-QUANTILE REGRESSION (QQR)
# ======================================================================
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from scipy.stats import ttest_rel

st.header("📈 Quantile-on-Quantile Regression (QQR) - Econometric Implementation")

# === User input
q_y = st.selectbox("Dependent variable (Y)", numeric_cols, index=0, key="qqr_y2")
q_x = st.selectbox("Independent variable (X)", [c for c in numeric_cols if c != q_y], index=0, key="qqr_x2")

# Detect panel dimension if any
potential_panels = [col for col in df.columns if col.lower() in ["country", "id", "entity", "firm", "region"]]
panel_col = potential_panels[0] if potential_panels else None
selected_groups = None
if panel_col:
    all_groups = df[panel_col].dropna().unique().tolist()
    selected_groups = st.multiselect(f"Select {panel_col}", all_groups, default=all_groups[:3])

quantile_n = st.slider("Number of Quantiles", 5, 30, 10, key="qqr_quantiles2")
bandwidth = st.slider("Bandwidth for kernel weighting", 0.01, 0.5, 0.05, 0.01, key="qqr_bandwidth")

# ======================================================================
# Core QQR Function
# ======================================================================
def proper_qqr_analysis(y, x, bandwidth, quantile_n, title_suffix=""):
    """Full QQR estimation (Sim & Zhou, 2015) with fallback for non-weighted QR."""
    data = pd.concat([y, x], axis=1).dropna()
    if data.empty or len(data) < 20:
        st.warning(f"⚠️ Insufficient observations for QQR {title_suffix}")
        return None, None, None, None

    y_data, x_data = data.iloc[:, 0], data.iloc[:, 1]
    n = len(y_data)

    tau_quantiles = np.linspace(0.05, 0.95, quantile_n)
    theta_quantiles = np.linspace(0.05, 0.95, quantile_n)

    beta_matrix = np.full((len(tau_quantiles), len(theta_quantiles)), np.nan)
    t_stat_matrix = np.full_like(beta_matrix, np.nan)
    p_value_matrix = np.full_like(beta_matrix, np.nan)

    progress_bar = st.progress(0)
    status_text = st.empty()

    def kernel(u, h):
        return np.exp(-0.5 * (u / h) ** 2) / (h * np.sqrt(2 * np.pi))

    total_iterations = len(tau_quantiles) * len(theta_quantiles)
    current_iteration = 0

    for i, tau in enumerate(tau_quantiles):
        for j, theta in enumerate(theta_quantiles):
            try:
                x_theta = np.quantile(x_data, theta)
                weights = kernel(x_data - x_theta, h=bandwidth)
                effective_n = np.sum(weights > 1e-6)

                if effective_n < 10:
                    current_iteration += 1
                    progress_bar.progress(current_iteration / total_iterations)
                    continue

                X_design = sm.add_constant(x_data)

                # Weighted Quantile Regression (try first)
                try:
                    model = QuantReg(y_data, X_design)
                    result = model.fit(q=tau, weights=weights, max_iter=1000)
                except TypeError:
                    raise RuntimeError("Weighted QuantReg not supported in this statsmodels version")

                if len(result.params) > 1:
                    beta_matrix[i, j] = result.params[1]
                    t_stat_matrix[i, j] = getattr(result, "tvalues", [np.nan, np.nan])[1]
                    p_value_matrix[i, j] = getattr(result, "pvalues", [np.nan, np.nan])[1]

            except Exception as e:
                # fallback: local unweighted QR on top-K highest weights
                try:
                    K = max(30, int(0.2 * n))
                    idx_top = np.argsort(weights)[-K:]
                    x_local = x_data.iloc[idx_top]
                    y_local = y_data.iloc[idx_top]
                    X_local = sm.add_constant(x_local)
                    model_local = QuantReg(y_local, X_local).fit(q=tau, max_iter=1000)
                    if len(model_local.params) > 1:
                        beta_matrix[i, j] = model_local.params[1]
                        t_stat_matrix[i, j] = model_local.tvalues[1]
                        p_value_matrix[i, j] = model_local.pvalues[1]
                except Exception as e_local:
                    st.write(f"QQR failed at τ={tau:.2f}, θ={theta:.2f}: {repr(e_local)}")

            current_iteration += 1
            progress_bar.progress(current_iteration / total_iterations)
            status_text.text(f"Estimating QQR: {current_iteration}/{total_iterations}")

    progress_bar.empty()
    status_text.empty()
    return beta_matrix, t_stat_matrix, p_value_matrix, tau_quantiles

# ======================================================================
# Visualization + Robustness
# ======================================================================
def run_qqr_with_robustness(y, x, title_suffix=""):
    beta_matrix, t_stats, p_values, tau_quantiles = proper_qqr_analysis(y, x, bandwidth, quantile_n, title_suffix)

    if beta_matrix is None or np.all(np.isnan(beta_matrix)):
        st.warning("⚠️ No valid QQR coefficients estimated. Try larger bandwidth or fewer quantiles.")
        return

    theta_quantiles = np.linspace(0.05, 0.95, quantile_n)

    # --- Coefficient Heatmap
    st.subheader("📊 QQR Coefficient Heatmap")
    fig_hm = go.Figure(go.Heatmap(
        z=beta_matrix,
        x=[f"{q:.2f}" for q in theta_quantiles],
        y=[f"{q:.2f}" for q in tau_quantiles],
        colorscale="RdBu_r",
        colorbar_title="Coefficient",
        zmid=0
    ))
    fig_hm.update_layout(
        title=f"QQR Coefficients {title_suffix}",
        xaxis_title=f"{q_x} Quantiles (θ)",
        yaxis_title=f"{q_y} Quantiles (τ)",
        template="plotly_white",
        width=800,
        height=600
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # --- 3D Surface
    st.subheader("📈 QQR 3D Surface")
    fig_3d = go.Figure(data=[go.Surface(
        z=beta_matrix,
        x=theta_quantiles,
        y=tau_quantiles,
        colorscale="RdBu_r",
        showscale=True
    )])
    fig_3d.update_layout(
        scene=dict(
            xaxis_title=f"{q_x} Quantiles (θ)",
            yaxis_title=f"{q_y} Quantiles (τ)",
            zaxis_title="Coefficient"
        ),
        title=f"3D QQR Surface {title_suffix}",
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=50),
        template="plotly_white"
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # --- Significance Heatmap
    if p_values is not None:
        st.subheader("🔬 Statistically Significant (p < 0.05)")
        sig_mask = p_values < 0.05
        sig_z = np.where(sig_mask, beta_matrix, np.nan)
        fig_sig = go.Figure(go.Heatmap(
            z=sig_z,
            x=[f"{q:.2f}" for q in theta_quantiles],
            y=[f"{q:.2f}" for q in tau_quantiles],
            colorscale="RdBu_r",
            colorbar_title="Significant Coefficients",
            zmid=0
        ))
        fig_sig.update_layout(
            title=f"Significant QQR Coefficients {title_suffix}",
            template="plotly_white",
            width=800,
            height=600
        )
        st.plotly_chart(fig_sig, use_container_width=True)

    # --- Robustness: QQR vs QR
    st.subheader("🔍 Robustness Check: QQR vs Standard Quantile Regression")
    qqr_avg = np.nanmean(beta_matrix, axis=1)

    qr_coeff, ci_lo, ci_hi = [], [], []
    X_std = sm.add_constant(x)
    for tau in tau_quantiles:
        try:
            model = QuantReg(y, X_std).fit(q=tau)
            qr_coeff.append(model.params[1])
            ci = model.conf_int()
            ci_lo.append(ci.iloc[1, 0])
            ci_hi.append(ci.iloc[1, 1])
        except Exception:
            qr_coeff.append(np.nan)
            ci_lo.append(np.nan)
            ci_hi.append(np.nan)

    comparison_df = pd.DataFrame({
        "Quantile": tau_quantiles,
        "QQR_Avg": qqr_avg,
        "QR": qr_coeff,
        "CI_L": ci_lo,
        "CI_U": ci_hi
    })
    st.dataframe(comparison_df.round(4), use_container_width=True)

    # Plot comparison
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(x=tau_quantiles, y=qqr_avg, mode="lines+markers", name="QQR Avg", line=dict(color="blue")))
    fig_cmp.add_trace(go.Scatter(x=tau_quantiles, y=qr_coeff, mode="lines+markers", name="QR Coef", line=dict(color="red", dash="dash")))
    fig_cmp.add_trace(go.Scatter(
        x=tau_quantiles.tolist() + tau_quantiles[::-1].tolist(),
        y=ci_hi + ci_lo[::-1],
        fill="toself", fillcolor="rgba(255,0,0,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="QR 95% CI"
    ))
    fig_cmp.update_layout(
        title=f"Robustness: QQR vs QR {title_suffix}",
        xaxis_title="Quantile (τ)",
        yaxis_title="Coefficient",
        template="plotly_white",
        width=800,
        height=500
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Correlation metrics
    mask = ~np.isnan(qr_coeff) & ~np.isnan(qqr_avg)
    if np.sum(mask) > 2:
        corr = np.corrcoef(np.array(qr_coeff)[mask], qqr_avg[mask])[0, 1]
        t_stat, p_val = ttest_rel(np.array(qr_coeff)[mask], qqr_avg[mask])
        col1, col2, col3 = st.columns(3)
        col1.metric("Correlation (QQR vs QR)", f"{corr:.4f}")
        col2.metric("t-test p-value", f"{p_val:.4f}")
        col3.metric("Mean Diff", f"{np.nanmean(qqr_avg - np.array(qr_coeff)):.4f}")

    # --- Download
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        pd.DataFrame(beta_matrix, 
                     index=[f"τ={q:.2f}" for q in tau_quantiles],
                     columns=[f"θ={q:.2f}" for q in theta_quantiles]).to_excel(writer, sheet_name="QQR_Coefficients")
        comparison_df.to_excel(writer, sheet_name="QQR_QR_Comparison", index=False)

    st.download_button(
        label="📥 Download QQR Results (Excel)",
        data=excel_buf.getvalue(),
        file_name="qqr_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ======================================================================
# Run by group or overall
# ======================================================================
if st.button("Run Proper QQR Analysis", key="qqr_run2"):
    if panel_col and selected_groups:
        for grp in selected_groups:
            subset = df[df[panel_col] == grp]
            st.subheader(f"{panel_col}: {grp}")
            run_qqr_with_robustness(subset[q_y], subset[q_x], f"({grp})")
    else:
        run_qqr_with_robustness(df[q_y], df[q_x])


# ======================================================================
# 🟩 SECTION 13: MACHINE LEARNING FORECASTING (IMPROVED)
# ======================================================================
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

st.header("🤖 Machine Learning Forecasting (Prophet Model)")

st.markdown("""
**Prophet Model Features:**
- Handles missing data and outliers automatically
- Captures multiple seasonality patterns
- Allows external regressors (optional)
- Provides uncertainty intervals
""")

dep_forecast = st.selectbox("Select dependent variable (target for forecasting)", numeric_cols, index=0, key="forecast_dep")
regressors = st.multiselect("Select independent variables (optional regressors)", 
                            [c for c in numeric_cols if c != dep_forecast])

col_f1, col_f2 = st.columns(2)
with col_f1:
    periods = st.number_input("Forecast horizon (future periods)", min_value=7, max_value=1000, value=90)
with col_f2:
    train_test_split = st.slider("Training data percentage", min_value=50, max_value=95, value=80)

if st.button("Run Forecast"):
    try:
        # Prepare data
        df_prophet = df.reset_index()[[df.index.name, dep_forecast]].rename(
            columns={df.index.name: "ds", dep_forecast: "y"}
        )
        
        # Train/test split for validation
        split_idx = int(len(df_prophet) * train_test_split / 100)
        train_data = df_prophet.iloc[:split_idx]
        test_data = df_prophet.iloc[split_idx:]
        
        st.info(f"Training on {len(train_data)} observations, testing on {len(test_data)} observations")
        
        # Initialize and fit model
        model = Prophet(
            daily_seasonality=True if len(df_prophet) > 730 else False,
            weekly_seasonality=True if len(df_prophet) > 14 else False,
            yearly_seasonality=True if len(df_prophet) > 365 else False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        # Add regressors if selected
        if regressors:
            st.warning("⚠️ Note: Regressors require future values for forecasting. Using last known values.")
            for reg in regressors:
                train_data[reg] = df[reg].iloc[:split_idx].values
                model.add_regressor(reg)
        
        model.fit(train_data)
        
        # Make predictions on test set
        if regressors:
            future_test = test_data[['ds']].copy()
            for reg in regressors:
                future_test[reg] = df[reg].iloc[split_idx:split_idx+len(test_data)].values
        else:
            future_test = test_data[['ds']]
        
        forecast_test = model.predict(future_test)
        
        # Calculate accuracy metrics
        y_true = test_data['y'].values
        y_pred = forecast_test['yhat'].values[:len(y_true)]
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        st.subheader("📊 Forecast Accuracy Metrics (Test Set)")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("MAE (Mean Absolute Error)", f"{mae:.4f}")
        with col_m2:
            st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.4f}")
        with col_m3:
            st.metric("MAPE (Mean Absolute % Error)", f"{mape:.2f}%")
        
        # Full forecast including future periods
        future_full = model.make_future_dataframe(periods=int(periods), freq="D")
        
        if regressors:
            for reg in regressors:
                last_val = df[reg].iloc[-1]
                future_full[reg] = last_val
        
        forecast_full = model.predict(future_full)
        
        # Plot results
        st.subheader("📈 Forecast Plot (Train + Test + Future)")
        fig_f = go.Figure()
        
        # Historical data
        fig_f.add_trace(go.Scatter(
            x=train_data["ds"], 
            y=train_data["y"], 
            mode="lines", 
            name="Training Data",
            line=dict(color="blue")
        ))
        
        # Test data
        fig_f.add_trace(go.Scatter(
            x=test_data["ds"], 
            y=test_data["y"], 
            mode="lines", 
            name="Test Data (Actual)",
            line=dict(color="green")
        ))
        
        # Full forecast
        fig_f.add_trace(go.Scatter(
            x=forecast_full["ds"], 
            y=forecast_full["yhat"], 
            mode="lines", 
            name="Forecast",
            line=dict(color="red", dash="dash")
        ))
        
        # Confidence intervals
        fig_f.add_trace(go.Scatter(
            x=forecast_full["ds"], 
            y=forecast_full["yhat_upper"], 
            fill=None, 
            mode="lines",
            line_color="rgba(255,0,0,0.2)", 
            showlegend=False
        ))
        fig_f.add_trace(go.Scatter(
            x=forecast_full["ds"], 
            y=forecast_full["yhat_lower"], 
            fill='tonexty', 
            mode="lines",
            line_color="rgba(255,0,0,0.2)", 
            name="95% Confidence Interval"
        ))
        
        fig_f.update_layout(
            title=f"Forecast for {dep_forecast} (Prophet Model)",
            plot_bgcolor=bg_color, 
            paper_bgcolor=bg_color,
            xaxis_title="Date",
            yaxis_title=dep_forecast,
            hovermode="x unified"
        )
        st.plotly_chart(fig_f, use_container_width=True)
        
        # Components plot
        st.subheader("📉 Forecast Components (Trend & Seasonality)")
        fig_comp = model.plot_components(forecast_full)
        st.pyplot(fig_comp)
        
        # Future predictions table
        st.subheader("📋 Future Forecast Table")
        future_only = forecast_full[forecast_full['ds'] > df.index.max()][["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(min(30, int(periods)))
        future_only.columns = ["Date", "Forecast", "Lower Bound", "Upper Bound"]
        st.dataframe(future_only.round(4), use_container_width=True)
        
        # Download forecast
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            forecast_full[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_excel(
                writer, index=False, sheet_name="Forecast"
            )
        
        st.download_button(
            label="📥 Download Full Forecast (Excel)",
            data=excel_buf.getvalue(),
            file_name="prophet_forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Forecasting failed: {e}")
        import traceback
        st.code(traceback.format_exc())

# ======================================================================
# 🟩 SECTION 14: EXPORT PROCESSED DATA
# ======================================================================
st.header("💾 Download Processed / Filtered Data")
to_download = df.reset_index()
csv_bytes = to_download.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV of processed data", csv_bytes, 
                   file_name="processed_time_series.csv", mime="text/csv")

# ======================================================================
# 🟩 FOOTER & HELP
# ======================================================================
st.sidebar.header("Help & Documentation")
st.sidebar.markdown("""
**Key Improvements:**
- ✅ Password protection added
- ✅ QQR robustness check with QR comparison
- ✅ ACF/PACF plots for ARIMA modeling
- ✅ All lags tested in Granger causality
- ✅ Train/test split for forecasting
- ✅ Accuracy metrics (MAE, RMSE, MAPE)
- ✅ Residual diagnostics
- ✅ Normality tests
- ✅ STL decomposition option
- ✅ Ljung-Box test for autocorrelation

**Econometric Best Practices:**
- Always check stationarity before Granger tests
- Look for multiple significant lags
- Validate forecasts on test data
- Examine residual diagnostics
- Check robustness of QQR findings
""")

st.sidebar.markdown("Run locally: `streamlit run app.py`")

# End of file
