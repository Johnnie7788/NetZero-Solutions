#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import plotly.express as px

# App Title
st.set_page_config(page_title="Sustainability Dashboard", layout="wide")
st.title("Sustainability Strategy Dashboard")

# Sidebar for Navigation
st.sidebar.title("Navigation")
sections = [
    "Home", 
    "Visualize Project Progress", 
    "Compliance Review", 
    "Performance Metrics", 
    "Risk Analysis", 
    "Recommendations & Alerts", 
    "Generate Reports"
]
choice = st.sidebar.radio("Navigate to", sections)

# Sidebar for File Upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Sustainability Data (CSV)", type=["csv"])

# Load Data
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Data uploaded successfully!")
    
    # Ensure all required columns are present
    required_columns = [
        'Project_Name', 'Category', 'Current_Value', 'Target_Value', 'Progress_Status', 
        'Budget (€)', 'Expenditure (€)', 'CSRD_Compliance', 'Stakeholder_Impact', 
        'Environmental_Risks', 'Supply_Chain_Disruptions', 'Mitigation_Status', 
        'Alerts', 'Recommendations'
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
        data = None
else:
    data = None

# Home Section
if choice == "Home":
    st.header("Welcome to the Sustainability Dashboard")
    st.write("""
    This dashboard is designed to:
    - Visualize sustainability project progress.
    - Review compliance with CSRD and DNK standards.
    - Track performance metrics and project statuses.
    - Conduct risk analysis for sustainability efforts.
    - Provide granular recommendations and alerts.
    - Generate professional, exportable reports for ESG compliance.
    """)
    if data is None:
        st.info("Upload data from the sidebar to get started.")

# Visualize Project Progress
elif choice == "Visualize Project Progress" and data is not None:
    st.header("Visualize Sustainability Project Progress")
    project_filter = st.selectbox("Select a Project", data['Project_Name'].unique())
    filtered_data = data[data['Project_Name'] == project_filter]
    st.write(filtered_data)
    fig = px.bar(filtered_data, x="Category", y="Current_Value", color="Progress_Status")
    st.plotly_chart(fig, use_container_width=True)

# Compliance Review
elif choice == "Compliance Review" and data is not None:
    if 'CSRD_Compliance' in data.columns:
        st.header("Review Compliance with CSRD and DNK Standards")
        compliance_summary = data.groupby("CSRD_Compliance").size().reset_index(name='Count')
        fig = px.pie(compliance_summary, names='CSRD_Compliance', values='Count', title="CSRD Compliance Overview")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("CSRD_Compliance column is missing from the dataset.")

# Performance Metrics
elif choice == "Performance Metrics" and data is not None:
    if 'Stakeholder_Impact' in data.columns:
        st.header("Track Performance Metrics and Status")
        st.dataframe(data[[
            'Project_Name', 'Category', 'Current_Value', 'Target_Value', 
            'Progress_Status', 'Budget (€)', 'Expenditure (€)', 'Stakeholder_Impact'
        ]])
    else:
        st.error("Stakeholder_Impact column is missing from the dataset.")

# Risk Analysis
elif choice == "Risk Analysis" and data is not None:
    if all(col in data.columns for col in ['Environmental_Risks', 'Supply_Chain_Disruptions', 'Mitigation_Status']):
        st.header("Analyze Sustainability Risks")
        risk_summary = data[['Project_Name', 'Environmental_Risks', 'Supply_Chain_Disruptions', 'Mitigation_Status']]
        st.write(risk_summary)
    else:
        st.error("One or more risk analysis columns are missing from the dataset.")

# Recommendations & Alerts
elif choice == "Recommendations & Alerts" and data is not None:
    if all(col in data.columns for col in ['Alerts', 'Recommendations']):
        st.header("Recommendations and Alerts")
        st.write(data[['Project_Name', 'Alerts', 'Recommendations']])
    else:
        st.error("Alerts or Recommendations columns are missing from the dataset.")

# Generate Reports
elif choice == "Generate Reports" and data is not None:
    if all(col in data.columns for col in ['CSRD_Compliance', 'Stakeholder_Impact', 'Environmental_Risks', 'Recommendations']):
        st.header("Generate Exportable Reports")
        report_data = data[[
            'Project_Name', 'Category', 'Progress_Status', 'CSRD_Compliance', 
            'Stakeholder_Impact', 'Environmental_Risks', 'Recommendations'
        ]]
        report_csv = report_data.to_csv(index=False)
        st.download_button(label="Download Report", data=report_csv, file_name="Sustainability_Report.csv", mime="text/csv")
    else:
        st.error("One or more required columns for the report are missing from the dataset.")

# Handle Missing Data
if data is None:
    st.error("Please upload a CSV file with all required columns to proceed.")


