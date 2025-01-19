#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import plotly.graph_objects as go
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Streamlit Configuration
st.set_page_config(page_title="Net Zero Scenario Modeling Tool", layout="wide")

# Complete Recommendations Data
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
        _, emissions = calculate_emissions(current_emissions, target, time_horizon)
        data[label] = emissions
    return data

# Function: Generate Recommendations
def generate_recommendations(sector, challenges):
    sector_rec = recommendations.get(sector, "No recommendations available.")
    additional_insights = f"Challenges: {challenges}" if challenges else "No challenges provided."
    return sector_rec, additional_insights

# Function: Generate PDF Report
def generate_pdf_report(data):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    c.setFont("Helvetica", 12)

    # Title
    c.drawString(30, 750, "Net Zero Emission Report")
    c.setFont("Helvetica", 10)

    # Inputs
    c.drawString(30, 730, f"Current Emissions: {data['current_emissions']} million tonnes")
    c.drawString(30, 715, f"Reduction Target: {data['reduction_target']}%")
    c.drawString(30, 700, f"Time Horizon: {data['time_horizon']} years")
    c.drawString(30, 685, f"Focus Sector: {data['sector']}")

    # Emissions Data
    c.drawString(30, 660, "Projected Emissions Over Time:")
    y_position = 645
    for year, emission in zip(data["years"], data["emissions"]):
        c.drawString(30, y_position, f"Year {year}: {emission:.2f} million tonnes")
        y_position -= 15
        if y_position < 50:
            c.showPage()
            c.setFont("Helvetica", 10)
            y_position = 750

    # Insights
    c.drawString(30, y_position, "Recommendations and Insights:")
    y_position -= 15
    recommendations = data["insights"].split("\n")
    for line in recommendations:
        c.drawString(30, y_position, line)
        y_position -= 15
        if y_position < 50:
            c.showPage()
            c.setFont("Helvetica", 10)
            y_position = 750

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

# Function: Plot Emissions
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

# Sidebar Inputs
st.sidebar.header("Model Inputs")
current_emissions = st.sidebar.number_input("Current CO₂ Emissions (million tonnes):", min_value=0.0, value=100.0)
reduction_target = st.sidebar.slider("Target Reduction by 2050 (%):", min_value=0, max_value=100, value=50)
time_horizon = st.sidebar.slider("Time Horizon (Years):", min_value=1, max_value=50, value=30)
sector = st.sidebar.selectbox("Select Sector:", list(recommendations.keys()))
user_challenges = st.sidebar.text_area("Describe challenges:", "")

# Process Data
years, emissions = calculate_emissions(current_emissions, reduction_target, time_horizon)
sensitivity_data = sensitivity_analysis(current_emissions, reduction_target, time_horizon)
sector_rec, insights = generate_recommendations(sector, user_challenges)

# Display Results
st.header("Emission Reduction Pathway")
plot_emissions_interactive(years, emissions, sensitivity_data)

st.subheader("Recommendations")
st.write(f"**{sector}:** {sector_rec}")
st.write(insights)

# Generate and Provide PDF Report
pdf_data = {
    "current_emissions": current_emissions,
    "reduction_target": reduction_target,
    "time_horizon": time_horizon,
    "years": years,
    "emissions": emissions,
    "sector": sector,
    "insights": f"{sector_rec}\n{insights}"
}
pdf_report = generate_pdf_report(pdf_data)
st.download_button(
    label="Download Net Zero Report as PDF",
    data=pdf_report,
    file_name="net_zero_report.pdf",
    mime="application/pdf"
)

