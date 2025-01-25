#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Streamlit Configuration
st.set_page_config(page_title="Net Zero Scenario Modeling Tool", layout="wide")

# Updated Global Sector Recommendations
recommendations = {
    "Energy": (
        "Increase deployment of solar and wind power via tax incentives and subsidies. "
        "Phase out coal plants by introducing carbon taxes and supporting worker transitions. "
        "Fund decentralized renewable systems for rural areas."
    ),
    "Transport": (
        "Expand public transport networks like metros and electric buses. "
        "Provide subsidies and low-interest loans for electric vehicles. "
        "Develop nationwide EV charging infrastructure by 2030. "
        "Implement urban congestion charges to reduce fossil fuel use."
    ),
    "Industry": (
        "Offer grants to SMEs for adopting energy-efficient machinery. "
        "Encourage industrial symbiosis, where waste from one firm is reused by another. "
        "Mandate energy audits for large industries with compliance incentives."
    ),
    "Buildings": (
        "Fund retrofits for better insulation and replace inefficient windows. "
        "Require all new buildings to meet green certifications like LEED. "
        "Subsidize smart thermostats and energy-efficient lighting installations."
    ),
    "Agriculture": (
        "Support farmers adopting precision agriculture like soil sensors. "
        "Educate on no-till farming for better soil carbon storage. "
        "Introduce methane-reducing feed additives for livestock."
    ),
    "Land Use and Forestry": (
        "Provide incentives to conserve forests and reforest degraded land. "
        "Use satellite monitoring to track and prevent illegal logging. "
        "Collaborate with NGOs to promote sustainable forestry practices."
    ),
    "Marine Ecosystems": (
        "Restore mangroves and coral reefs through marine protected areas. "
        "Promote sustainable small-scale fisheries to protect biodiversity. "
        "Invest in seaweed farming for carbon sequestration."
    ),
    "Waste Management": (
        "Mandate recycling programs and separation of waste at source. "
        "Build anaerobic digestion plants for biogas from organic waste. "
        "Encourage reusable packaging and refillable product systems."
    ),
    "Circular Economy": (
        "Adopt producer responsibility systems for lifecycle management. "
        "Promote modular and recyclable product designs. "
        "Fund innovations in recycling technologies and material recovery."
    ),
    "Energy Storage and Grid Innovation": (
        "Research advanced battery technologies like solid-state batteries. "
        "Deploy time-of-use tariffs to manage peak electricity demand. "
        "Roll out smart grid projects for energy distribution efficiency."
    ),
    "Behavioral and Consumer Changes": (
        "Launch campaigns on energy and water conservation benefits. "
        "Incentivize plant-based diets and eco-friendly consumer choices. "
        "Subsidize household solar and energy-efficient appliances."
    )
}

# Function: Collect User Inputs
def get_user_inputs():
    st.sidebar.header("Model Inputs")
    current_emissions = st.sidebar.number_input("Current CO₂ Emissions (million tonnes):", min_value=0.0, value=100.0, step=0.1)
    reduction_target = st.sidebar.slider("Target Reduction by 2050 (%):", min_value=0, max_value=100, value=50)
    time_horizon = st.sidebar.slider("Time Horizon (Years):", min_value=1, max_value=50, value=30)
    sector = st.sidebar.selectbox("Select Sector:", list(recommendations.keys()))
    user_challenges = st.sidebar.text_area("Describe your organization's challenges:", "e.g., funding constraints, lack of expertise")
    return current_emissions, reduction_target, time_horizon, sector, user_challenges

# Function: Calculate Emissions
def calculate_emissions(current_emissions, reduction_target, time_horizon):
    years = np.arange(2023, 2023 + time_horizon + 1)
    reduction_per_year = (reduction_target / 100) * current_emissions / time_horizon
    emissions = [current_emissions - i * reduction_per_year for i in range(len(years))]
    return years, emissions

# Function: Sensitivity Analysis
def sensitivity_analysis(current_emissions, reduction_target, time_horizon):
    scenarios = [
        (reduction_target - 10, "Pessimistic"),
        (reduction_target, "Baseline"),
        (reduction_target + 10, "Optimistic")
    ]
    data = {}
    for target, label in scenarios:
        if label != "Baseline":  # Exclude baseline scenario to avoid redundancy
            _, emissions = calculate_emissions(current_emissions, target, time_horizon)
            data[label] = emissions
    return data

# Function: Generate Recommendations
def generate_recommendations(sector, challenges):
    st.subheader("Sector-Specific Recommendations")
    st.write(f"**{sector}:** {recommendations.get(sector, 'No recommendations available.')}")

    if challenges:
        st.subheader("Additional Insights from Challenges")
        st.write(f"Key user-provided challenges: {challenges}")
        st.write("Consider aligning solutions with these challenges for a customized approach.")

# Function: Generate PDF Report Using ReportLab
def generate_pdf_report(current_emissions, reduction_target, time_horizon, emissions, years, sector, insights):
    """Generates a detailed PDF report using ReportLab with robust text wrapping."""
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    c.setFont("Helvetica", 12)

    # Title
    c.drawString(30, 750, "Net Zero Emission Report")
    c.setFont("Helvetica", 10)

    # Inputs
    c.drawString(30, 730, f"Current Emissions: {current_emissions} million tonnes")
    c.drawString(30, 715, f"Reduction Target: {reduction_target}%")
    c.drawString(30, 700, f"Time Horizon: {time_horizon} years")
    c.drawString(30, 685, f"Focus Sector: {sector}")

    # Emissions Data
    c.drawString(30, 660, "Projected Emissions Over Time:")
    y_position = 645
    for year, emission in zip(years, emissions):
        c.drawString(30, y_position, f"Year {year}: {emission:.2f} million tonnes")
        y_position -= 15
        if y_position < 50:
            c.showPage()
            c.setFont("Helvetica", 10)
            y_position = 750

    # Insights Section
    c.drawString(30, y_position, "Insights and Recommendations:")
    y_position -= 15

    # Text Wrapping for Recommendations
    def draw_wrapped_text(text, x, y, max_width, line_height):
        words = text.split()
        line = ""
        for word in words:
            if c.stringWidth(line + " " + word, "Helvetica", 10) <= max_width:
                line += " " + word if line else word
            else:
                c.drawString(x, y, line)
                line = word
                y -= line_height
                if y < 50:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = 750
        if line:
            c.drawString(x, y, line)
            y -= line_height
        return y

    # Wrap insights text
    max_width = 500  # Adjust to fit page width
    y_position = draw_wrapped_text(insights, 30, y_position, max_width, 15)

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

# Function: Plot Emissions with Interactive Visualizations
def plot_emissions_interactive(years, emissions, sensitivity_data):
    fig = go.Figure()

    # Add main projection
    fig.add_trace(go.Scatter(
        x=years, y=emissions, mode='lines+markers', name="Baseline Projection",
        line=dict(color='green')
    ))

    # Add sensitivity scenarios
    for label, data in sensitivity_data.items():
        fig.add_trace(go.Scatter(
            x=years, y=data, mode='lines', name=f"{label} Scenario",
            line=dict(dash='dot')
        ))

    fig.update_layout(
        title="CO₂ Emission Reduction Pathway",
        xaxis_title="Year",
        yaxis_title="CO₂ Emissions (million tonnes)",
        template="plotly_white",
        legend_title="Scenarios"
    )
    st.plotly_chart(fig)

# Main Application
st.title("Net Zero Scenario Modeling Tool")
st.markdown("This tool helps organizations set targets and model Net Zero scenarios while addressing unique challenges.")

# Get User Inputs
current_emissions, reduction_target, time_horizon, sector, user_challenges = get_user_inputs()

# Calculate Emissions
years, emissions = calculate_emissions(current_emissions, reduction_target, time_horizon)

# Sensitivity Analysis
sensitivity_data = sensitivity_analysis(current_emissions, reduction_target, time_horizon)

# Display Emission Reduction Pathway
st.header("Emission Reduction Pathway")
plot_emissions_interactive(years, emissions, sensitivity_data)

# Generate Recommendations
generate_recommendations(sector, user_challenges)

# Compile Insights for Report
sector_recommendation = recommendations.get(sector, "No recommendations available.")
insight_text = f"Sector: {sector}\nRecommendations: {sector_recommendation}\nUser Challenges: {user_challenges}"

# Generate and Provide PDF Download
pdf_report = generate_pdf_report(current_emissions, reduction_target, time_horizon, emissions, years, sector, insight_text)
st.download_button(
    label="Download Net Zero Report as PDF",
    data=pdf_report,
    file_name="net_zero_report.pdf",
    mime="application/pdf"
)

