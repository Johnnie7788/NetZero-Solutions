#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO

# Page Configuration
st.set_page_config(page_title="🌍 Climate Strategy Dashboard", layout="wide")

# App Header
st.markdown(
    """
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 18px;
        color: #34495E;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    <div class="main-header">Climate Strategy Dashboard</div>
    <div class="sub-header">Track emissions, monitor compliance, analyze risks, and optimize sustainability strategies</div>
    """,
    unsafe_allow_html=True
)

# Sidebar for Data Upload
st.sidebar.header("📂 Upload Your Dataset")
st.sidebar.markdown(
    """
    **Required Columns in CSV File:**

    | Column Name                   | Description                                                          |
    |-------------------------------|----------------------------------------------------------------------|
    | **Region**                   | Geographic area (e.g., Europe, Asia, North America).                |
    | **Sector**                   | Industry sector (e.g., Energy, Transport, Agriculture).             |
    | **Carbon_Emissions_MT**      | Carbon emissions in million tons.                                   |
    | **Energy_Efficiency_%**      | Energy efficiency as a percentage.                                  |
    | **Compliance_Status**        | Compliance status (e.g., Compliant, Non-Compliant).                 |
    | **Climate_Risk_Score**       | Climate risk score (scale: 1-5).                                     |
    | **SDG_Alignment_Score**      | Alignment with SDGs (scale: 0-100).                                  |
    | **Renewable_Energy_Adoption_%** | Renewable energy adoption percentage.                              |
    | **Policy_Impact_Reduction_%**| Impact of policies in percentage reduction.                         |
    | **Climate_Adaptation_Index** | Vulnerability to climate risks (scale: 0-100).                      |
    | **Estimated_Financial_Impact (€)** | Projected financial cost of climate policies (in Euros).         |
    | **Biodiversity_Impact_Score**| Impact of operations on biodiversity (scale: 1-5).                  |
    | **Circularity_Score**        | Resource efficiency and recycling rate (scale: 0-100).              |
    """
)

data = None
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Dataset Uploaded Successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {e}")
else:
    st.sidebar.warning("📄 Please upload a valid dataset to start analysis.")

# If Data Exists
if data is not None:
    st.subheader("📊 Dataset Overview")
    st.dataframe(data)

    # Key Metrics
    st.subheader("📌 Key Sustainability Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_emissions = data["Carbon_Emissions_MT"].sum()
        st.metric("Total Carbon Emissions (MT)", f"{total_emissions:.2f}")
    with col2:
        avg_efficiency = data["Energy_Efficiency_%"].mean()
        st.metric("Avg Energy Efficiency (%)", f"{avg_efficiency:.2f}")
    with col3:
        compliance_rate = (data["Compliance_Status"] == "Compliant").mean() * 100
        st.metric("Compliance Rate (%)", f"{compliance_rate:.2f}")
    with col4:
        avg_risk_score = data["Climate_Risk_Score"].mean()
        st.metric("Avg Climate Risk Score", f"{avg_risk_score:.2f}")

    # Visualizations
    st.subheader("📈 Visual Analysis")
    st.markdown("### Carbon Emissions by Region")
    fig, ax = plt.subplots()
    sns.barplot(data=data, x="Region", y="Carbon_Emissions_MT", palette="viridis", ax=ax)
    ax.set_title("Carbon Emissions by Region", fontsize=14)
    st.pyplot(fig)

    st.markdown("### Renewable Energy Adoption vs. Carbon Emissions")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x="Renewable_Energy_Adoption_%", y="Carbon_Emissions_MT", hue="Region", s=100, palette="coolwarm", ax=ax)
    ax.set_title("Renewable Energy vs Carbon Emissions", fontsize=14)
    st.pyplot(fig)

    st.markdown("### Climate Adaptation Index by Region")
    fig, ax = plt.subplots()
    sns.barplot(data=data, x="Region", y="Climate_Adaptation_Index", palette="cubehelix", ax=ax)
    ax.set_title("Climate Adaptation Index by Region", fontsize=14)
    st.pyplot(fig)

    # Benchmarking Tool
    st.subheader("📊 Benchmarking Tool")
    benchmark_metric = st.selectbox("Select a Metric to Benchmark:", data.columns[2:])
    benchmark_fig, benchmark_ax = plt.subplots()
    sns.boxplot(data=data, x="Region", y=benchmark_metric, palette="Set2", ax=benchmark_ax)
    benchmark_ax.set_title(f"Benchmarking {benchmark_metric} by Region", fontsize=14)
    st.pyplot(benchmark_fig)

    # Scenario Analysis with Forecasting
    st.subheader("🔮 Scenario Analysis and Forecasting")
    reduction_percent = st.slider("Select Emission Reduction Target (%)", min_value=0, max_value=50, step=5)
    st.write(f"Impact of {reduction_percent}% Reduction:")
    data["Reduced_Emissions"] = data["Carbon_Emissions_MT"] * (1 - reduction_percent / 100)

    # Forecast future emissions
    forecast_years = st.slider("Select Forecasting Horizon (years):", min_value=1, max_value=10, step=1)
    future_emissions = data["Reduced_Emissions"] * np.exp(-0.05 * forecast_years)  # Example decay model
    forecast_df = data[["Region", "Sector", "Carbon_Emissions_MT", "Reduced_Emissions"]].copy()
    forecast_df[f"Emissions_{forecast_years}_Years"] = future_emissions

    # Forecast Visualization
    st.markdown(f"### Forecasted Emissions Over {forecast_years} Years")
    fig, ax = plt.subplots()
    sns.lineplot(data=forecast_df, x="Region", y=f"Emissions_{forecast_years}_Years", hue="Sector", marker="o", ax=ax)
    ax.set_title(f"Forecasted Emissions by Region and Sector ({forecast_years} Years)", fontsize=14)
    st.pyplot(fig)

    # Circularity Score Analysis
    st.markdown("### Circularity Score by Sector")
    fig, ax = plt.subplots()
    sns.barplot(data=data, x="Sector", y="Circularity_Score", palette="muted", ax=ax)
    ax.set_title("Circularity Score by Sector", fontsize=14)
    st.pyplot(fig)

    # Custom Recommendations
    st.subheader("📋 Customized Recommendations")
    recommendations = []
    if data["Renewable_Energy_Adoption_%"].mean() < 50:
        recommendations.append("Increase renewable energy adoption to above 50% to meet global standards.")
    if data["Circularity_Score"].mean() < 70:
        recommendations.append("Focus on improving circularity practices, such as recycling and reuse strategies.")
    if data["Climate_Risk_Score"].mean() > 3.5:
        recommendations.append("Implement stronger risk mitigation strategies for climate-sensitive sectors.")

    if recommendations:
        st.markdown("### Key Recommendations:")
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.write("All metrics are within optimal ranges. Great job!")

    # Downloadable Report
    st.subheader("📥 Downloadable Report")
    report = StringIO()
    forecast_df.to_csv(report, index=False)
    st.download_button("Download Report as CSV", data=report.getvalue(), file_name="Enhanced_Climate_Strategy_Report.csv", mime="text/csv")

else:
    st.info("Upload a dataset to begin analysis.")

