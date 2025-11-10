# ======================================================================
# 🟩 ADVANCED TIME SERIES ANALYZER – Econometrically Improved Version
# ======================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm
from prophet import Prophet
import io

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
# 🟩 IMPROVED GRANGER CAUSALITY TEST WITH STATIONARITY CHECK
# ======================================================================
st.header("🔁 Improved Granger Causality Test")

def check_stationarity(series, max_lags=5):
    """Check if series is stationary using ADF test"""
    try:
        result = adfuller(series.dropna(), maxlag=max_lags, autolag='AIC')
        return result[1] <= 0.05  # p-value <= 0.05 indicates stationarity
    except:
        return False

def find_optimal_lag(granger_results, max_lag):
    """Find optimal lag using AIC from Granger test results"""
    aic_values = []
    for lag in range(1, max_lag + 1):
        aic = granger_results[lag][2]  # AIC is the third element in the tuple
        aic_values.append((lag, aic))
    
    if aic_values:
        optimal_lag = min(aic_values, key=lambda x: x[1])[0]
        return optimal_lag
    return max_lag

# 🟩 User input selections
dependent_var = st.selectbox("Select Dependent Variable", numeric_cols)
independent_vars = st.multiselect("Select Independent Variable(s)", [c for c in numeric_cols if c != dependent_var])
max_lag = st.selectbox("Select Maximum Lag", list(range(1, 11)), index=4)

if st.button("Run Improved Granger Causality Test"):
    try:
        # 🟩 Stationarity Check
        st.subheader("📊 Stationarity Check (Required for Valid Test)")
        stationary_results = []
        
        # Check dependent variable
        y_stationary = check_stationarity(df[dependent_var])
        stationary_results.append({
            "Variable": dependent_var,
            "Stationary": "✅ Yes" if y_stationary else "❌ No",
            "Recommendation": "OK" if y_stationary else "Use first differences"
        })
        
        # Check independent variables
        for indep in independent_vars:
            x_stationary = check_stationarity(df[indep])
            stationary_results.append({
                "Variable": indep,
                "Stationary": "✅ Yes" if x_stationary else "❌ No",
                "Recommendation": "OK" if x_stationary else "Use first differences"
            })
        
        stationary_df = pd.DataFrame(stationary_results)
        st.dataframe(stationary_df, use_container_width=True)
        
        # Warn if any series are non-stationary
        all_stationary = all([check_stationarity(df[var]) for var in [dependent_var] + independent_vars])
        if not all_stationary:
            st.warning("⚠️ Some series are non-stationary. Granger causality results may be invalid!")
            st.info("💡 Consider using first differences of the non-stationary series for valid testing.")
        
        # 🟩 Run Granger Tests
        results_list = []
        
        for indep in independent_vars:
            temp_df = df[[dependent_var, indep]].dropna()
            
            # Run Granger test for multiple lags
            gc_res = grangercausalitytests(temp_df, maxlag=max_lag, verbose=False)
            
            # Find optimal lag using AIC
            optimal_lag = find_optimal_lag(gc_res, max_lag)
            
            # Get results for optimal lag
            f_test = gc_res[optimal_lag][0]['ssr_ftest'][0]
            p_value = gc_res[optimal_lag][0]['ssr_ftest'][1]
            
            # Determine significance
            significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
            
            results_list.append({
                "Hypothesis Tested": f"{indep} ➜ {dependent_var}",
                "Optimal Lag": optimal_lag,
                "F-statistic": round(f_test, 4),
                "P-Value": round(p_value, 4),
                "Significance": significance,
                "Conclusion": "Causal" if p_value < 0.05 else "No causal"
            })
            
            # Reverse direction test
            temp_df_rev = df[[indep, dependent_var]].dropna()
            gc_rev = grangercausalitytests(temp_df_rev, maxlag=max_lag, verbose=False)
            optimal_lag_rev = find_optimal_lag(gc_rev, max_lag)
            f_test_rev = gc_rev[optimal_lag_rev][0]['ssr_ftest'][0]
            p_value_rev = gc_rev[optimal_lag_rev][0]['ssr_ftest'][1]
            significance_rev = "***" if p_value_rev < 0.01 else "**" if p_value_rev < 0.05 else "*" if p_value_rev < 0.1 else ""
            
            results_list.append({
                "Hypothesis Tested": f"{dependent_var} ➜ {indep}",
                "Optimal Lag": optimal_lag_rev,
                "F-statistic": round(f_test_rev, 4),
                "P-Value": round(p_value_rev, 4),
                "Significance": significance_rev,
                "Conclusion": "Causal" if p_value_rev < 0.05 else "No causal"
            })

        # 🟩 Display Results
        gc_df = pd.DataFrame(results_list)
        st.subheader("📊 Granger Causality Results")
        st.dataframe(gc_df, use_container_width=True)
        
        # 🟩 Download Results
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            stationary_df.to_excel(writer, index=False, sheet_name="StationarityCheck")
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
# 🟩 PROPER QUANTILE REGRESSION (REPLACING INCORRECT QQR)
# ======================================================================
st.header("📈 Proper Quantile Regression")

q_y = st.selectbox("Dependent variable (Y)", numeric_cols, index=0, key="qr_y")
q_x = st.selectbox("Independent variable (X)", [c for c in numeric_cols if c != q_y], index=0, key="qr_x")

quantiles = st.slider("Select Quantiles", 0.05, 0.95, (0.25, 0.5, 0.75), 0.05, key="qr_quantiles")
if isinstance(quantiles, float):
    quantiles = [quantiles]

if st.button("Run Quantile Regression"):
    try:
        # Prepare data
        data = df[[q_y, q_x]].dropna()
        if len(data) < 10:
            st.warning("Insufficient data points for quantile regression")
            st.stop()
        
        y = data[q_y]
        X = sm.add_constant(data[q_x])  # Add constant for intercept
        
        # Run quantile regression for each quantile
        results = {}
        coef_data = []
        
        for q in quantiles:
            model = QuantReg(y, X).fit(q=q)
            results[q] = model
            
            # Store coefficient information
            coef_data.append({
                'Quantile': q,
                'Intercept': model.params[0],
                'Coefficient': model.params[1],
                'P-Value': model.pvalues[1],
                'CI_Lower': model.conf_int()[0][1],
                'CI_Upper': model.conf_int()[1][1]
            })
        
        # Create results dataframe
        results_df = pd.DataFrame(coef_data)
        
        # Display results
        st.subheader("📊 Quantile Regression Results")
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # 🟩 Plot 1: Coefficient Plot across Quantiles
        fig_coef = go.Figure()
        
        fig_coef.add_trace(go.Scatter(
            x=results_df['Quantile'],
            y=results_df['Coefficient'],
            mode='lines+markers',
            name='Coefficient',
            line=dict(width=3)
        ))
        
        # Add confidence intervals
        fig_coef.add_trace(go.Scatter(
            x=results_df['Quantile'].tolist() + results_df['Quantile'].tolist()[::-1],
            y=results_df['CI_Upper'].tolist() + results_df['CI_Lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI'
        ))
        
        fig_coef.update_layout(
            title=f"Quantile Regression Coefficients: {q_x} → {q_y}",
            xaxis_title="Quantile",
            yaxis_title="Coefficient Value",
            showlegend=True
        )
        st.plotly_chart(fig_coef, use_container_width=True)
        
        # 🟩 Plot 2: Quantile Regression Lines
        fig_lines = go.Figure()
        
        # Scatter plot of actual data
        fig_lines.add_trace(go.Scatter(
            x=data[q_x], y=data[q_y],
            mode='markers',
            name='Actual Data',
            marker=dict(size=4, opacity=0.6)
        ))
        
        # Add regression lines for each quantile
        x_range = np.linspace(data[q_x].min(), data[q_x].max(), 100)
        
        for q in quantiles:
            model = results[q]
            y_pred = model.params[0] + model.params[1] * x_range
            
            fig_lines.add_trace(go.Scatter(
                x=x_range, y=y_pred,
                mode='lines',
                name=f'Q{q}',
                line=dict(width=2)
            ))
        
        fig_lines.update_layout(
            title=f"Quantile Regression Lines: {q_x} → {q_y}",
            xaxis_title=q_x,
            yaxis_title=q_y
        )
        st.plotly_chart(fig_lines, use_container_width=True)
        
        # 🟩 Download Results
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            results_df.to_excel(writer, index=False, sheet_name="QuantileRegression")
        
        st.download_button(
            label="📥 Download Quantile Regression Results",
            data=excel_buf.getvalue(),
            file_name="quantile_regression_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"Error in quantile regression: {e}")

# ======================================================================
# 🟩 IMPROVED PROPHET FORECASTING WITH BETTER REGRESSOR HANDLING
# ======================================================================
st.header("🤖 Improved ML Forecasting (Prophet)")

# Select dependent and independent variables
dep_forecast = st.selectbox("Select dependent variable", numeric_cols, index=0, key="forecast_dep")
regressors = st.multiselect("Select independent variables (regressors)", 
                           [c for c in numeric_cols if c != dep_forecast])

# Forecast settings
periods = st.number_input("Forecast horizon (days)", min_value=7, max_value=1000, value=90)
include_confidence = st.checkbox("Include confidence intervals", value=True)
confidence_level = st.slider("Confidence level", 0.80, 0.99, 0.95)

if st.button("Run Improved Forecast"):
    try:
        # Prepare data for Prophet
        df_prophet = df.reset_index()[[df.index.name, dep_forecast] + regressors]
        df_prophet = df_prophet.rename(columns={df.index.name: "ds", dep_forecast: "y"})
        
        # Initialize and configure model
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            interval_width=confidence_level
        )
        
        # Add regressors if specified
        for reg in regressors:
            model.add_regressor(reg)
        
        # Fit model
        model.fit(df_prophet)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=int(periods), freq="D")
        
        # 🟩 IMPROVED: Better future regressor handling
        if regressors:
            for reg in regressors:
                # Use simple trend projection for future regressors
                last_value = df_prophet[reg].iloc[-1]
                trend = df_prophet[reg].diff().mean()
                
                if np.isnan(trend) or trend == 0:
                    # If no clear trend, use last value
                    future[reg] = last_value
                else:
                    # Project trend forward
                    future_days = len(future) - len(df_prophet)
                    future_values = [last_value + trend * (i+1) for i in range(future_days)]
                    future.loc[future.index[-future_days:], reg] = future_values
                    future[reg].fillna(method='ffill', inplace=True)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # 🟩 Plot 1: Main forecast plot
        st.subheader("📈 Forecast Plot")
        fig_forecast = model.plot(forecast)
        plt.title(f"Forecast for {dep_forecast}")
        st.pyplot(fig_forecast)
        
        # 🟩 Plot 2: Components
        st.subheader("🔧 Forecast Components")
        fig_components = model.plot_components(forecast)
        st.pyplot(fig_components)
        
        # 🟩 Display forecast summary
        st.subheader("📊 Forecast Summary")
        forecast_summary = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
        forecast_summary['ds'] = forecast_summary['ds'].dt.strftime('%Y-%m-%d')
        st.dataframe(forecast_summary.round(3), use_container_width=True)
        
        # 🟩 Model diagnostics
        st.subheader("📋 Model Diagnostics")
        
        # Calculate performance metrics on historical data
        historical = forecast[forecast['ds'].isin(df_prophet['ds'])]
        if not historical.empty:
            actual = df_prophet['y'].values
            predicted = historical['yhat'].values
            
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted)**2))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            metrics_df = pd.DataFrame({
                'Metric': ['MAE', 'RMSE', 'MAPE (%)'],
                'Value': [round(mae, 4), round(rmse, 4), round(mape, 2)]
            })
            
            st.dataframe(metrics_df, use_container_width=True)
        
        # 🟩 Download forecast
        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast",
            data=csv,
            file_name=f"forecast_{dep_forecast}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Forecasting failed: {e}")

# ======================================================================
# 🟩 BASIC VISUALIZATIONS (Unchanged from your original)
# ======================================================================
st.header("📊 Basic Visualizations & Statistics")

# Display basic stats
st.subheader("Descriptive Statistics")
st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)

# Time series plot
if len(numeric_cols) > 0:
    st.subheader("Time Series Plot")
    selected_series = st.multiselect("Select series to plot", numeric_cols, default=numeric_cols[:2])
    
    if selected_series:
        fig_ts = go.Figure()
        for col in selected_series:
            fig_ts.add_trace(go.Scatter(
                x=df.index, y=df[col],
                mode='lines',
                name=col
            ))
        
        fig_ts.update_layout(
            title="Time Series Plot",
            xaxis_title="Date",
            yaxis_title="Value"
        )
        st.plotly_chart(fig_ts, use_container_width=True)

# Correlation heatmap
st.subheader("Correlation Heatmap")
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

# ======================================================================
# 🟩 FOOTER
# ======================================================================
st.sidebar.header("Help & Information")
st.sidebar.markdown("""
**Improvements Made:**
- ✅ Proper Granger causality with stationarity checks
- ✅ Real quantile regression (replaced incorrect QQR)
- ✅ Better Prophet forecasting with improved regressor handling
- ✅ Model diagnostics and validation
""")

st.sidebar.markdown("**Requirements:** `streamlit pandas numpy matplotlib seaborn plotly statsmodels openpyxl prophet scipy`")
