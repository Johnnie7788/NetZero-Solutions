#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import heapq
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
from scipy.spatial.distance import euclidean
import seaborn as sns
from datetime import datetime, timedelta


# ---- APP CONFIG ----
st.set_page_config(page_title="AI-SCUTS", layout="wide")
st.title("AI-Powered Supply Chain & Unified Transformation Suite (AI-SCUTS)")

# ---- FILE UPLOAD SECTION ----
st.sidebar.header("Upload Your Supply Chain Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Data successfully loaded!")
else:
    st.warning("Please upload a dataset to proceed.")
    st.stop()

# ---- 1. SUPPLY CHAIN RISK & RESILIENCE ANALYSIS ----
st.subheader("📊 Supply Chain Risk & Resilience Analysis")

# Calculate a Weighted Risk Score
data["Risk_Score"] = (
    (data["CO2_Emissions"] / data["CO2_Emissions"].max()) * 40 +
    (data["Lead_Time_Days"] / data["Lead_Time_Days"].max()) * 30 +
    ((100 - data["Reliability_Score"]) / 100) * 20 +
    ((data["Stock_Variability"]) / data["Stock_Variability"].max()) * 10
)

# Dynamic Risk Classification
def dynamic_risk_classification(score):
    if score > 75:
        return "Severe Risk"
    elif score > 50:
        return "High Risk"
    elif score > 30:
        return "Moderate Risk"
    else:
        return "Low Risk"

data["Risk_Category"] = data["Risk_Score"].apply(dynamic_risk_classification)

# Visualization: Supplier Risk Distribution
st.subheader("📡 Supplier Risk Score Visualization")
fig = px.bar(data, x="Supplier", y="Risk_Score", color="Risk_Category", title="Supplier Risk Score")
st.plotly_chart(fig)

# Add Supplier Resilience Strategies
def recommend_resilience_strategy(risk_category):
    if risk_category == "Severe Risk":
        return "Diversify supplier network, reduce lead time, adopt local sourcing"
    elif risk_category == "High Risk":
        return "Optimize logistics, renegotiate contracts for reliability"
    elif risk_category == "Moderate Risk":
        return "Implement real-time tracking, improve demand planning"
    else:
        return "Monitor periodically, maintain strong supplier relationships"

data["Resilience_Strategy"] = data["Risk_Category"].apply(recommend_resilience_strategy)
st.write("📌 Supplier Risk & Resilience Recommendations:")
st.dataframe(data[["Supplier", "Risk_Category", "Resilience_Strategy"]])

# Real-Time Alerts for High-Risk Suppliers
high_risk_suppliers = data[data["Risk_Score"] > 75][["Supplier", "Risk_Score"]]
if not high_risk_suppliers.empty:
    st.warning("⚠️ High-Risk Suppliers Detected! Consider Immediate Action.")
    st.dataframe(high_risk_suppliers)


# ---- 2. SCENARIO-BASED SALES & OPERATIONS PLANNING ----
st.subheader("📅 Scenario-Based Sales & Operations Planning")

demand_factor = st.slider("Adjust Demand Factor", 0.5, 2.0, 1.0)
data["Optimized_Demand"] = data["Lead_Time_Days"] * demand_factor

# AI-Based Demand Forecasting using Random Forest
predictors = ["Lead_Time_Days", "Cost_Per_Unit"]
X_train, X_test, y_train, y_test = train_test_split(data[predictors], data["Optimized_Demand"], test_size=0.2, random_state=42)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
data["AI_Forecasted_Demand"] = rf.predict(data[predictors])

# Scenario Simulation (Best & Worst Case)
data["Worst_Case_Demand"] = data["Optimized_Demand"] * 0.8  # 20% drop
data["Best_Case_Demand"] = data["Optimized_Demand"] * 1.2  # 20% increase

# ARIMA Forecasting
st.subheader("📈 Forecasted Demand (ARIMA Model)")
arima_model = ARIMA(data["Optimized_Demand"], order=(2,1,2))
model_fit = arima_model.fit()
forecast = model_fit.forecast(steps=5)

# LSTM Model for Time-Series Forecasting
st.subheader("📊 AI-Based LSTM Demand Prediction")
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train, y_train, epochs=50, batch_size=16)
data["LSTM_Predicted_Demand"] = lstm_model.predict(data[predictors])

# Visualization: ARIMA vs. AI vs. Scenarios
st.subheader("📡 Multi-Scenario Demand Forecasting")
fig = go.Figure()
fig.add_trace(go.Scatter(y=forecast, mode='lines', name='ARIMA Forecast'))
fig.add_trace(go.Scatter(y=data["AI_Forecasted_Demand"], mode='lines', name='AI Forecast'))
fig.add_trace(go.Scatter(y=data["Best_Case_Demand"], mode='lines', name='Best Case'))
fig.add_trace(go.Scatter(y=data["Worst_Case_Demand"], mode='lines', name='Worst Case'))
fig.add_trace(go.Scatter(y=data["LSTM_Predicted_Demand"], mode='lines', name='LSTM Forecast'))
st.plotly_chart(fig)

# Risk Analysis for Potential Stock Shortages
data["Risk_Level"] = data.apply(lambda row: "High" if row["AI_Forecasted_Demand"] > row["Current_Stock"] else "Low", axis=1)
st.write("⚠️ Supply Chain Risk Assessment:")
st.dataframe(data[["Supplier", "AI_Forecasted_Demand", "Current_Stock", "Risk_Level"]])


# ---- 3. SUPPLY CHAIN DIGITAL TWIN VISUALIZATION ----
st.subheader("🌍 Supply Chain Digital Twin")

# Build Graph Network
G = nx.Graph()
for i, row in data.iterrows():
    G.add_node(row["Supplier"], pos=(row["Latitude"], row["Longitude"]))
    for j, r in data.iterrows():
        if row["Supplier"] != r["Supplier"]:
            distance = geodesic((row["Latitude"], row["Longitude"]), (r["Latitude"], r["Longitude"])).km
            G.add_edge(row["Supplier"], r["Supplier"], weight=distance)

# Add Edge Labels (Distance Display)
fig, ax = plt.subplots()
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, ax=ax)
edge_labels = {(row["Supplier"], r["Supplier"]): f"{geodesic((row["Latitude"], row["Longitude"]), (r["Latitude"], r["Longitude"])).km:.1f} km" 
                for i, row in data.iterrows() for j, r in data.iterrows() if row["Supplier"] != r["Supplier"]}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
st.pyplot(fig)

# Shortest Path Calculation (Dijkstra Algorithm)
def dijkstra_shortest_path(graph, start, end):
    return nx.shortest_path(graph, source=start, target=end, weight='weight')

st.subheader("🔍 Optimized Route Finder")
start_supplier = st.selectbox("Select Start Supplier", data["Supplier"].unique())
end_supplier = st.selectbox("Select Destination Supplier", data["Supplier"].unique())
shortest_path = dijkstra_shortest_path(G, start_supplier, end_supplier)
st.write("🛣 Shortest Path: ", " → ".join(shortest_path))

# Interactive Plotly Visualization of Digital Twin
st.subheader("📡 Interactive Supply Chain Network")
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

fig = go.Figure()
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="gray")))
for node in G.nodes():
    x, y = pos[node]
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=node, marker=dict(size=10, color="blue")))
st.plotly_chart(fig)

# Add Traffic Delay Data for Real-Time Insights
data["Traffic_Delay"] = np.random.randint(5, 30, size=len(data))  # Simulating traffic delays in minutes
data["Adjusted_Travel_Time"] = data["Traffic_Delay"] + data["Lead_Time_Days"]
st.write("🚦 Adjusted Travel Time with Traffic Delays:")
st.dataframe(data[["Supplier", "Lead_Time_Days", "Traffic_Delay", "Adjusted_Travel_Time"]])


# ---- 4. REAL-TIME COST ANALYSIS & SUPPLIER PERFORMANCE RANKING ----
st.subheader("💰 Supplier Performance Ranking")

# Normalize Cost, CO2, and Lead Time to avoid skewed rankings
data["Cost_Score"] = 1 - (data["Cost_Per_Unit"] - data["Cost_Per_Unit"].min()) / (data["Cost_Per_Unit"].max() - data["Cost_Per_Unit"].min())
data["CO2_Score"] = 1 - (data["CO2_Emissions"] - data["CO2_Emissions"].min()) / (data["CO2_Emissions"].max() - data["CO2_Emissions"].min())
data["Lead_Time_Score"] = 1 - (data["Lead_Time_Days"] - data["Lead_Time_Days"].min()) / (data["Lead_Time_Days"].max() - data["Lead_Time_Days"].min())

# Classify Supplier Risk Level
def classify_risk(row):
    if row["Reliability_Score"] < 50 or row["Lead_Time_Days"] > data["Lead_Time_Days"].quantile(0.75):
        return "High Risk"
    else:
        return "Low Risk"
data["Risk_Category"] = data.apply(classify_risk, axis=1)

# Include Supplier Order Fulfillment Rate
data["Fulfillment_Rate"] = data["Orders_Fulfilled"] / data["Total_Orders"]

# Weighted Supplier Performance Score
data["Performance_Score"] = (
    (data["Reliability_Score"] * 0.35) +
    (data["Cost_Score"] * 0.25) +
    (data["CO2_Score"] * 0.15) +
    (data["Lead_Time_Score"] * 0.15) +
    (data["Fulfillment_Rate"] * 0.10)
)

# Display the ranking table
st.write("📌 Supplier Performance & Risk Analysis:")
st.dataframe(data[["Supplier", "Performance_Score", "Risk_Category", "Fulfillment_Rate"]].sort_values("Performance_Score", ascending=False))

# Visualization: Supplier Performance Bar Chart
fig = px.bar(data.sort_values("Performance_Score", ascending=False),
             x="Supplier", y="Performance_Score",
             color="Risk_Category",
             title="Supplier Performance & Risk Rating")
st.plotly_chart(fig)


# ---- 5. AI-BASED COST & INVENTORY OPTIMIZATION ----
st.subheader("📉 AI-Based Cost & Inventory Optimization")

def calculate_eoq(demand, ordering_cost, holding_cost):
    return np.sqrt((2 * demand * ordering_cost) / holding_cost)

# Calculate EOQ (Economic Order Quantity)
data["EOQ"] = calculate_eoq(data["Optimized_Demand"], 50, 5)

# Calculate Safety Stock (20% of expected demand during lead time)
data["Safety_Stock"] = data["Lead_Time_Days"] * data["Optimized_Demand"] * 0.2

# Calculate Reorder Point (ROP)
data["Reorder_Point"] = (data["Lead_Time_Days"] * data["Optimized_Demand"]) + data["Safety_Stock"]

# Calculate Inventory Turnover Ratio (Efficiency of inventory usage)
data["Inventory_Turnover"] = data["Optimized_Demand"] / data["Current_Stock"]

# AI-Based Demand Forecasting using Random Forest
rf = RandomForestRegressor(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    data[["Lead_Time_Days", "Cost_Per_Unit"]], data["Optimized_Demand"], test_size=0.2, random_state=42)
rf.fit(X_train, y_train)
data["AI_Forecasted_Demand"] = rf.predict(data[["Lead_Time_Days", "Cost_Per_Unit"]])

# Calculate Inventory Holding Cost (Assumed 15% of total inventory value)
data["Holding_Cost"] = data["EOQ"] * data["Cost_Per_Unit"] * 0.15

st.write("📌 Cost & Inventory Optimization Insights:")
st.dataframe(data[["Supplier", "EOQ", "Safety_Stock", "Reorder_Point", "Inventory_Turnover", "AI_Forecasted_Demand", "Holding_Cost"]])

# Visualization: EOQ vs. Inventory Costs
fig = px.line(data, x="Supplier", y=["EOQ", "Holding_Cost"], title="EOQ vs. Inventory Costs")
st.plotly_chart(fig)


# ---- 6. MACHINE LEARNING-BASED WAREHOUSE EFFICIENCY MODELING ----
st.subheader("🏢 Machine Learning-Based Warehouse Efficiency Modeling")

# Calculate Space Utilization (%)
data["Space_Utilization"] = (data["Current_Stock"] / data["Warehouse_Capacity"]) * 100

def warehouse_strategy(utilization):
    if utilization < 50:
        return "Reallocate inventory"
    elif utilization > 90:
        return "Expand storage capacity"
    else:
        return "Optimized"

data["Warehouse_Strategy"] = data["Space_Utilization"].apply(warehouse_strategy)

# Apply K-Means Clustering
X_warehouse = data[["Warehouse_Capacity", "Current_Stock"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_warehouse)
kmeans = KMeans(n_clusters=2, random_state=42)
data["Efficiency_Cluster"] = kmeans.fit_predict(X_scaled)

# Label Clusters
cluster_labels = {0: "Overstocked", 1: "Efficient"}
data["Efficiency_Cluster_Label"] = data["Efficiency_Cluster"].map(cluster_labels)

# Calculate Warehouse Productivity Score
data["Warehouse_Score"] = (
    (data["Stock_Turnover"] * 0.4) +
    (data["Space_Utilization"] * 0.3) +
    (data["Order_Fulfillment_Speed"] * 0.3)
)

st.write("📌 Warehouse Efficiency Insights:")
st.dataframe(data[["Supplier", "Space_Utilization", "Efficiency_Cluster_Label", "Warehouse_Strategy", "Warehouse_Score"]])

# Visualization: Warehouse Utilization Heatmap
fig = px.bar(
    data, x="Supplier", y="Space_Utilization", color="Efficiency_Cluster_Label", title="Warehouse Utilization")
st.plotly_chart(fig)


# ---- 7. SUSTAINABILITY & GREEN SUPPLY CHAIN ANALYSIS ----
st.subheader("🌱 Sustainability & Green Supply Chain Analytics")

# Define a list of certified green suppliers (this could be fetched from an external source in real applications)
certified_suppliers = ["Supplier_A", "Supplier_C", "Supplier_F"]  # Example list of certified suppliers
data["Green_Certified"] = data["Supplier"].apply(lambda x: "Yes" if x in certified_suppliers else "No")

# Identify low-emission suppliers
low_emission_suppliers = data[data["CO2_Emissions"] < data["CO2_Emissions"].mean()]
st.write("✅ Recommended Green Suppliers:")
st.dataframe(low_emission_suppliers[["Supplier", "CO2_Emissions", "Green_Certified"]])

# Calculate Sustainability Score (Weighted)
data["Sustainability_Score"] = (
    (1 / data["CO2_Emissions"]) * 0.5 + 
    (1 / data["Distance_km"]) * 0.3 + 
    (data["Green_Certified"].apply(lambda x: 1 if x == "Yes" else 0)) * 0.2
)

# Recommend CO₂ Reduction Strategies
def recommend_co2_strategy(co2_emissions):
    if co2_emissions > data["CO2_Emissions"].quantile(0.75):
        return "Switch to Renewable Energy"
    elif co2_emissions > data["CO2_Emissions"].median():
        return "Optimize Transportation Routes"
    else:
        return "Sustainable Packaging & Materials"

data["CO2_Reduction_Strategy"] = data["CO2_Emissions"].apply(recommend_co2_strategy)
st.write("📌 CO₂ Reduction Recommendations:")
st.dataframe(data[["Supplier", "CO2_Emissions", "CO2_Reduction_Strategy"]])

# Alternative Transport Mode Recommendations
def recommend_transport(distance):
    if distance > 1000:
        return "Rail Freight"
    elif distance > 500:
        return "Electric Trucks"
    else:
        return "Local Distribution"

data["Recommended_Transport"] = data["Distance_km"].apply(recommend_transport)
st.write("🚛 Recommended Transport Methods:")
st.dataframe(data[["Supplier", "Distance_km", "Recommended_Transport"]])

# Visualization: Potential CO₂ Reduction Impact
fig = px.bar(data, x="Supplier", y="CO2_Emissions", color="Green_Certified", title="CO₂ Reduction Potential by Supplier")
st.plotly_chart(fig)


# ---- 8. LOGISTICS & TRANSPORTATION OPTIMIZATION ----
st.subheader("🚚 Logistics & Transportation Optimization")

# Define Dijkstra's Algorithm for shortest path calculation
def dijkstra(graph, start):
    queue = []
    heapq.heappush(queue, (0, start))
    distances = {node: float("inf") for node in graph.nodes}
    distances[start] = 0
    previous_nodes = {node: None for node in graph.nodes}

    while queue:
        current_distance, current_node = heapq.heappop(queue)
        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]['weight']
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    return distances, previous_nodes

# Convert Data to a NetworkX Graph
G = nx.Graph()
for _, row in data.iterrows():
    G.add_node(row["Supplier"], pos=(row["Latitude"], row["Longitude"]))

for _, row in data.iterrows():
    for _, r in data.iterrows():
        if row["Supplier"] != r["Supplier"]:
            distance = geodesic((row["Latitude"], row["Longitude"]), (r["Latitude"], r["Longitude"])).km
            G.add_edge(row["Supplier"], r["Supplier"], weight=distance)

# Select starting supplier for route optimization
start_supplier = st.selectbox("Select Start Supplier", data["Supplier"].unique(), key="start_supplier_logistics")
distances, previous_nodes = dijkstra(G, start_supplier)

# Display shortest routes
st.subheader("📍 Shortest Routes to Other Suppliers")
st.dataframe(pd.DataFrame(distances.items(), columns=["Supplier", "Distance (km)"]).sort_values("Distance (km)"))

# Visualizing optimized logistics routes (Only One Instance Now)
fig, ax = plt.subplots()
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", ax=ax)
nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
nx.draw_networkx_edges(G, pos, edge_color="black", ax=ax)
st.pyplot(fig)  # Only One Visualization Instance


# ---- 9. MANAGEMENT & REPORTING DASHBOARD ----
st.subheader("📊 Management & Reporting Dashboard")
st.write("🔹 **Top 3 Insights**")
st.write("1️⃣ Highest Performing Supplier: ", data.loc[data["Performance_Score"].idxmax(), "Supplier"])
st.write("2️⃣ Greenest Supplier (Lowest CO2): ", data.loc[data["CO2_Emissions"].idxmin(), "Supplier"])
st.write("3️⃣ Optimal Warehouse Efficiency Score: ", data["Efficiency_Cluster"].value_counts().idxmax())

st.subheader("📌 Strategic Recommendations")
st.write("✔ Optimize inventory levels based on EOQ calculations to minimize costs.")
st.write("✔ Prioritize suppliers with high reliability and low CO2 emissions.")
st.write("✔ Improve warehouse efficiency by reallocating high-volume stock strategically.")
st.write("✔ Reduce transportation costs by leveraging optimized routing solutions.")
st.write("✔ Monitor and adjust S&OP forecasts dynamically to prevent stock imbalances.")

