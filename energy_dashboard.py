
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
from prophet import Prophet

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="UWI Cavehill Energy Dashboard",
                   page_icon="📶", layout="wide")
st.title("📶 UWI Cavehill Energy Dashboard")
st.markdown("---")

# -------------------------
# Helper functions
# -------------------------
def simulate_data():
    """Return a simulated multi-building monthly dataset (3 years)."""
    np.random.seed(42)
    buildings = ["Sydney Martin Library", "FST Building", "Sir Philip Sherlock Hall", "Admin Building"]
    commodities = ["Electricity", "Water", "Gas"]
    date_rng = pd.date_range(start="2020-01-01", end=pd.Timestamp.today(), freq="M") # 5 years monthly
    rows = []
    for b in buildings:
        for c in commodities:
            # Create seasonality for electricity, less for water/gas
            base_usage = {"Electricity": 4000, "Water": 500, "Gas": 1500}[c]
            for d in date_rng:
                season = 1 + 0.2 * np.sin(2 * np.pi * (d.month / 12.0))
                usage = base_usage * season * np.random.uniform(0.85, 1.15)
                cost_per_unit = {
                    "Electricity": 0.678,  # BBD per kWh
                    "Water": 7.78,  #BBD per metre cubed
                    "Gas": 0.08  # Placeholder
                }[c]
                cost = usage * cost_per_unit * np.random.uniform(0.95, 1.1)
                rows.append({
                    "Date": d,
                    "Building": b,
                    "Commodity": c,
                    "Usage": usage,
                    "Cost": cost
                })
    df_sim = pd.DataFrame(rows)
    return df_sim

def compute_kpis(df_rolling):
    total_cost = df_rolling["Cost"].sum()
    total_usage = df_rolling["Usage"].sum()
    co2_est = df_rolling["Usage"].sum() * 0.0007
    return total_cost, total_usage, co2_est

def percent_change(current, previous):
    if previous == 0:
        return np.nan
    return (current - previous) / previous * 100

# -------------------------
# Data input (upload or simulate)
# -------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional) — columns: Date, Building, Commodity, Usage, Cost", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=["Date"])
    # Basic validation & cleanup
    if "Building" not in df.columns: df["Building"] = "Unknown"
    if "Commodity" not in df.columns: df["Commodity"] = "Unknown"
    if "Usage" not in df.columns: df["Usage"] = df.get("Usage", np.nan)
    if "Cost" not in df.columns: df["Cost"] = df.get("Cost", np.nan)
else:
    df = simulate_data()

# Derived columns
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["MonthName"] = df["Date"].dt.strftime("%b")
df = df.sort_values("Date")

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters")
buildings = ["All"] + sorted(df["Building"].unique().tolist())
selected_building = st.sidebar.selectbox("Building", buildings, index=0)
commodities = sorted(df["Commodity"].unique().tolist())
selected_commodities = st.sidebar.multiselect("Commodity (multi)", commodities, default=commodities)
# time range filter (min/max date)
min_date = df["Date"].min()
max_date = df["Date"].max()
start_date, end_date = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())

# apply filters
mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date) & (df["Commodity"].isin(selected_commodities))
if selected_building != "All":
    mask &= (df["Building"] == selected_building)
filtered_df = df.loc[mask].copy()

# If filtered_df is empty show a message
if filtered_df.empty:
    st.warning("No data for the selected filters. Try widening the date range or selecting different commodities/buildings.")
    st.stop()

date_start_display = filtered_df["Date"].min().strftime("%b %Y")
date_end_display = filtered_df["Date"].max().strftime("%b %Y")

st.sidebar.markdown("- Electricity: BBD 0.678/kWh\n- Water: BBD 7.78/m³\n- Gas: BBD 0.08/unit (placeholder)")
# -------------------------
# Rolling 12 months and previous 12 months
# -------------------------
latest_date = filtered_df["Date"].max()
rolling_12_start = (latest_date - pd.DateOffset(months=12)) + pd.Timedelta(days=1)
prev_12_start = rolling_12_start - pd.DateOffset(months=12)

df_12m = filtered_df[filtered_df["Date"] >= rolling_12_start]
df_prev_12m = filtered_df[(filtered_df["Date"] >= prev_12_start) & (filtered_df["Date"] < rolling_12_start)]

# -------------------------
# KPIs (within the last 12 months)
# -------------------------
st.subheader(f"Overview — {date_start_display} to {date_end_display}")
total_cost, total_usage, co2_est = compute_kpis(df_12m)
prev_cost, prev_usage, prev_co2 = compute_kpis(df_prev_12m)

col1, col2, col3, col4 = st.columns([1,1,1,1])
col1.metric("Total Cost (last 12m, BBD)", f"${total_cost:,.0f}", f"{percent_change(total_cost, prev_cost):+.1f}%")
col2.metric("Total Usage (kWh)", f"{total_usage:,.0f}", f"{percent_change(total_usage, prev_usage):+.1f}%")
col3.metric("Estimated CO₂ (tons)", f"{co2_est/1000:,.2f}", f"{percent_change(co2_est, prev_co2):+.1f}%")
col4.download_button("Download filtered CSV", data=filtered_df.to_csv(index=False), file_name="filtered_energy.csv", mime="text/csv")

st.markdown("---")

# -------------------------
#-------------------real time gauges-------------------------
# -------------------------
st.subheader("Real-Time Performance Gauges")

g1, g2, g3 = st.columns(3)

import plotly.graph_objects as go

# Gauge 1 – Monthly Cost
fig_cost = go.Figure(go.Indicator(
    mode="gauge+number",
    value=total_cost,
    title={'text': "Total Cost (12m)"},
    gauge={'axis': {'range': [None, total_cost * 1.2]}},
))
g1.plotly_chart(fig_cost, use_container_width=True)

# Gauge 2 – Usage
fig_usage = go.Figure(go.Indicator(
    mode="gauge+number",
    value=total_usage,
    title={'text': "Energy Usage (kWh)"},
    gauge={'axis': {'range': [None, total_usage * 1.2]}},
))
g2.plotly_chart(fig_usage, use_container_width=True)

# Gauge 3 – CO2
fig_co2 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=co2_est/1000,
    title={'text': "CO₂ (tons)"},
    gauge={'axis': {'range': [None, (co2_est/1000) * 1.2]}},
))
g3.plotly_chart(fig_co2, use_container_width=True)


# -------------------------
#------------------charts for distribution and trends-------------------
# -------------------------
st.subheader(f"Trends & Distribution ({date_start_display} — {date_end_display})")


left, right = st.columns([2,3])

with left:
    st.markdown(f"**Cost share By Commodity ({date_start_display} — {date_end_display})**")

    pie = (
        alt.Chart(df_12m)
        .mark_arc(innerRadius=40)
        .encode(
            theta=alt.Theta("sum(Cost):Q", title="Cost"),
            color=alt.Color("Commodity:N", legend=alt.Legend(orient="bottom")),
            tooltip=[alt.Tooltip("Commodity:N"), alt.Tooltip("sum(Cost):Q", title="Total Cost")]
        )
        .properties(height=300)
    )
    st.altair_chart(pie, use_container_width=True)

    st.markdown(f"**Monthly totals ({date_start_display} — {date_end_display})**")

    area = (
        alt.Chart(df_12m.groupby(["Date","Commodity"], as_index=False).sum())
        .mark_area(opacity=0.8)
        .encode(
            x="Date:T",
            y="sum(Cost):Q",
            color="Commodity:N",
            tooltip=["Date:T", "Commodity:N", "sum(Cost):Q"]
        )
        .properties(height=300)
    )
    st.altair_chart(area, use_container_width=True)

with right:
    st.markdown(f"**Monthly Cost by Building ({date_start_display} — {date_end_display})**")


    # Sum only numeric columns to avoid datetime errors
    pivot = (
        df_12m
        .groupby(["MonthName", "Building"], as_index=False)[["Cost", "Usage"]]
        .sum()
    )

    # Ensure Month order for correct display
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot["MonthName"] = pd.Categorical(pivot["MonthName"], categories=month_order, ordered=True)

    # Plot
    small = (
        alt.Chart(pivot)
        .mark_bar()
        .encode(
            x=alt.X("MonthName:N", sort=month_order),
            y=alt.Y("Cost:Q", title="Monthly Cost (BBD)"),
            color="Building:N",
            tooltip=["Building", "MonthName", "Cost"]
        )
        .properties(height=400)
    )
    st.altair_chart(small, use_container_width=True)


# -------------------------
# Yearly Comparison (plotly)
# -------------------------
st.subheader("Yearly Comparison")

# Select only numeric columns before summing
numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns

yearly = (
    filtered_df
    .groupby([filtered_df["Date"].dt.year.rename("Year"), "Commodity"], as_index=False)[numeric_cols]
    .sum()
)

fig = px.bar(yearly, x="Year", y="Cost")
fig.update_layout(
    title=f"Yearly Comparison ({date_start_display} — {date_end_display})"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -------------------------
# Forecasting with Prophet
# -------------------------
st.subheader("Forecasting")

forecast_commodity = st.selectbox("Select commodity for forecast", options=sorted(filtered_df["Commodity"].unique()), index=0)
forecast_horizon = st.selectbox("Forecast horizon (months)", options=[6, 12, 24], index=1)

fc_df = filtered_df[filtered_df["Commodity"] == forecast_commodity].groupby("Date", as_index=False).sum()[["Date","Cost"]].rename(columns={"Date":"ds","Cost":"y"})
if len(fc_df) < 12:
    st.info("Not enough data to generate a reliable forecast. Need at least 12 months for Prophet.")
else:
    m = Prophet()
    m.fit(fc_df)
    future = m.make_future_dataframe(periods=forecast_horizon, freq="M")
    fcst = m.predict(future)
    # Plot using plotly
    fig_fc = px.line(fcst, x="ds", y="yhat", labels={"ds":"Date","yhat":"Predicted Cost (BBD)"}, title=f"{forecast_commodity} Forecast ({forecast_horizon} months beyond {date_end_display})")

    fig_fc.add_scatter(x=fc_df["ds"], y=fc_df["y"], mode="markers", name="Historical")
    st.plotly_chart(fig_fc, use_container_width=True)

st.markdown("---")

# -------------------------
# Scenario simulation
# -------------------------
st.subheader("Scenario Simulation")

reduction = st.slider("Set energy reduction target (%) - applies to Usage", min_value=0, max_value=50, value=10)
simulated_reduction_cost = df_12m.copy()
simulated_reduction_cost["Usage_adj"] = simulated_reduction_cost["Usage"] * (1 - reduction/100)
# record same cost-per-unit ratio from original
simulated_reduction_cost["Cost_adj"] = simulated_reduction_cost["Cost"] * (simulated_reduction_cost["Usage_adj"] / simulated_reduction_cost["Usage"])
sim_total_cost = simulated_reduction_cost["Cost_adj"].sum()
savings = total_cost - sim_total_cost
sim_co2 = simulated_reduction_cost["Usage_adj"].sum() * 0.000601

st.write(f"With a **{reduction}%** reduction in usage you would save approximately **${savings:,.0f}** annually and reduce CO₂ by **{(co2_est - sim_co2)/1000:,.2f}** tons.")
st.progress(reduction / 100)
st.markdown("---")
st.caption("© 2025 UWI Cavehill Energy Dashboard | Built with Streamlit ⚡")
