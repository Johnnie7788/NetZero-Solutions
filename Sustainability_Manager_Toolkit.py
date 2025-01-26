#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import plotly.express as px

# App Title and Configuration
st.set_page_config(page_title="Sustainability Manager Toolkit", layout="wide")
st.title("Sustainability Manager Toolkit")

# Sidebar Navigation
st.sidebar.title("Navigation")
sections = [
    "Home",
    "Analyze Sustainability Data",
    "Collaborate for Service Improvement",
    "Project Management",
    "ESG Score Tracker",
    "Supply Chain Sustainability",
    "Compliance & Regulations",
    "Stakeholder Engagement",
]
choice = st.sidebar.radio("Navigate to", sections)

# Sidebar File Upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Sustainability Data (CSV)", type=["csv"])

# Load and Validate Data
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Data uploaded successfully!")
    st.write("### Dataset Preview")
    st.dataframe(data.head())  # Display the dataset for verification
    st.write("### Dataset Columns")
    st.write(data.columns.tolist())  # Display columns to check alignment
else:
    st.error("Please upload a valid CSV file to proceed.")
    data = None

# Define Section Functions
def home_section():
    st.header("Welcome to the Sustainability Manager Toolkit")
    st.write("""
    This application is designed to:
    - Analyze sustainability-related data for challenges and opportunities.
    - Collaborate with teams to improve service levels and sustainability positions.
    - Track project progress and manage initiatives.
    - Enhance ESG scores through compliance and risk mitigation.
    - Evaluate supply chain sustainability and align with corporate goals.
    - Educate stakeholders and generate actionable insights.
    """)
    if data is None:
        st.info("Upload data from the sidebar to get started.")

def analyze_data_section():
    if data is not None:
        st.header("Analyze Sustainability Data")
        st.write("This section helps identify challenges and opportunities in the dataset.")
        if all(col in data.columns for col in ["Current_Value", "Target_Value", "Progress_Status"]):
            # Calculate progress gaps
            data["Gap"] = data["Target_Value"] - data["Current_Value"]
            challenges = data[data["Gap"] > 20]
            st.subheader("Projects with Significant Gaps:")
            st.dataframe(challenges)
            
            # Bar chart for opportunities
            fig = px.bar(data, x="Project_Name", y="Gap", color="Progress_Status", title="Gap Analysis")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("The required columns (Current_Value, Target_Value, Progress_Status) are missing.")
    else:
        st.error("Please upload a dataset to view this section.")

def collaborate_section():
    if data is not None:
        st.header("Collaborate for Service Improvement")
        if all(col in data.columns for col in ["Stakeholder_Impact", "Recommendations"]):
            st.write("This section highlights stakeholder impact and recommendations for collaboration.")
            
            # Stakeholder impact summary
            impact_summary = data.groupby("Stakeholder_Impact").size().reset_index(name="Count")
            st.subheader("Stakeholder Impact Summary:")
            st.dataframe(impact_summary)
            
            # Recommendations visualization
            st.subheader("Recommendations for Collaboration:")
            fig = px.bar(data, x="Project_Name", y="Stakeholder_Impact", color="Recommendations", title="Collaboration Recommendations")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("The required columns (Stakeholder_Impact, Recommendations) are missing.")
    else:
        st.error("Please upload a dataset to view this section.")

def project_management_section():
    if data is not None:
        st.header("Project Management")
        if "Project_Name" in data.columns and "Progress_Status" in data.columns:
            project_filter = st.selectbox("Select a Project", data["Project_Name"].unique())
            filtered_data = data[data["Project_Name"] == project_filter]
            if not filtered_data.empty:
                st.write(filtered_data)
                fig = px.bar(filtered_data, x="Category", y="Current_Value", color="Progress_Status")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected project.")
        else:
            st.error("Required columns for Project Management are missing: Project_Name, Progress_Status.")
    else:
        st.error("Please upload a dataset to view this section.")

def esg_score_tracker_section():
    if data is not None:
        st.header("ESG Score Tracker")
        if "CSRD_Compliance" in data.columns:
            compliance_summary = data.groupby("CSRD_Compliance").size().reset_index(name="Count")
            st.write(compliance_summary)
            fig = px.pie(compliance_summary, names="CSRD_Compliance", values="Count", title="CSRD Compliance Overview")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("The dataset does not contain the required column: CSRD_Compliance.")
    else:
        st.error("Please upload a dataset to view this section.")

def supply_chain_section():
    if data is not None:
        st.header("Supply Chain Sustainability")
        if all(col in data.columns for col in ["Environmental_Risks", "Supply_Chain_Disruptions"]):
            st.dataframe(data[["Project_Name", "Environmental_Risks", "Supply_Chain_Disruptions"]])
        else:
            st.warning("Required columns for Supply Chain Sustainability are missing.")
    else:
        st.error("Please upload a dataset to view this section.")

def compliance_section():
    if data is not None:
        st.header("Compliance & Regulations")
        if all(col in data.columns for col in ["CSRD_Compliance", "Stakeholder_Impact"]):
            compliance_data = data[["Project_Name", "CSRD_Compliance", "Stakeholder_Impact"]]
            st.dataframe(compliance_data)
        else:
            st.warning("Required columns for Compliance & Regulations are missing.")
    else:
        st.error("Please upload a dataset to view this section.")

def stakeholder_section():
    if data is not None:
        st.header("Stakeholder Engagement")
        if all(col in data.columns for col in ["Alerts", "Recommendations"]):
            st.dataframe(data[["Project_Name", "Alerts", "Recommendations"]])
        else:
            st.warning("Required columns for Stakeholder Engagement are missing.")
    else:
        st.error("Please upload a dataset to view this section.")

# Render Section Based on Selection
if choice == "Home":
    home_section()
elif choice == "Analyze Sustainability Data":
    analyze_data_section()
elif choice == "Collaborate for Service Improvement":
    collaborate_section()
elif choice == "Project Management":
    project_management_section()
elif choice == "ESG Score Tracker":
    esg_score_tracker_section()
elif choice == "Supply Chain Sustainability":
    supply_chain_section()
elif choice == "Compliance & Regulations":
    compliance_section()
elif choice == "Stakeholder Engagement":
    stakeholder_section()

# Handle Missing Data
if data is None:
    st.warning("Make sure your dataset contains all required columns for full functionality.")

