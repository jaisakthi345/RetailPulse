Python 3.14.5 (v3.14.5:5607950ef23, May 10 2026, 07:38:09) [Clang 21.0.0 (clang-2100.0.123.102)] on darwin
Enter "help" below or click "Help" above for more information.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prophet import Prophet

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="RetailPulse - AI Analytics", layout="wide")
st.title("📊 RetailPulse - AI Powered Retail Analytics Dashboard")
st.markdown("End-to-End Data Science platform for demand prediction & customer insights.")

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", [
    "📁 Upload Dataset", "📊 Sales Analytics", "👥 Customer Segmentation", 
    "📈 Demand Forecasting", "⚠️ Churn Prediction", "📦 Inventory Optimization"
])

# --- 3. DATA UPLOAD & PROCESSING ---
if page == "📁 Upload Dataset":
    st.header("Upload Retail Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
        st.session_state["data"] = df
        st.success("Dataset ready!")
        st.dataframe(df.head())

if "data" in st.session_state:
    df = st.session_state["data"]
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# --- 4. SALES ANALYTICS ---
if page == "📊 Sales Analytics" and "data" in st.session_state:
    st.header("Executive Sales Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"₹ {df['TotalPrice'].sum():,.0f}")
    col2.metric("Total Orders", df["InvoiceNo"].nunique())
    col3.metric("Total Customers", df["CustomerID"].nunique())

    st.subheader("Daily Sales Trend")
    fig, ax = plt.subplots()
    df.groupby("InvoiceDate")["TotalPrice"].sum().plot(ax=ax)
    st.pyplot(fig)

# --- 5. CUSTOMER SEGMENTATION ---
if page == "👥 Customer Segmentation" and "data" in st.session_state:
    st.header("RFM Behavioral Segmentation")
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "count",
        "TotalPrice": "sum"
    }).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency", "TotalPrice": "Monetary"})
...     
...     rfm["Cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(StandardScaler().fit_transform(rfm))
...     
...     fig, ax = plt.subplots()
...     sns.scatterplot(x="Recency", y="Monetary", hue="Cluster", data=rfm, ax=ax)
...     st.pyplot(fig)
... 
... # --- 6. DEMAND FORECASTING ---
... if page == "📈 Demand Forecasting" and "data" in st.session_state:
...     st.header("30-Day Demand Forecast")
...     daily_sales = df.groupby("InvoiceDate")["TotalPrice"].sum().reset_index().rename(columns={"InvoiceDate": "ds", "TotalPrice": "y"})
...     model = Prophet().fit(daily_sales)
...     forecast = model.predict(model.make_future_dataframe(periods=30))
...     
...     st.pyplot(model.plot(forecast))
... 
... # --- 7. CHURN PREDICTION ---
... if page == "⚠️ Churn Prediction" and "data" in st.session_state:
...     st.header("At-Risk Customers (>90 Days Inactive)")
...     snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
...     churn_df = ((snapshot_date - df.groupby("CustomerID")["InvoiceDate"].max()).dt.days > 90).reset_index().rename(columns={"InvoiceDate": "Churn"})
...     
...     fig, ax = plt.subplots()
...     churn_df["Churn"].value_counts().plot(kind="bar", ax=ax, title="Active vs Churned")
...     st.pyplot(fig)
... 
... # --- 8. INVENTORY OPTIMIZATION ---
... if page == "📦 Inventory Optimization" and "data" in st.session_state:
...     st.header("Automated Restock Recommendations")
...     daily_sales = df.groupby("InvoiceDate")["TotalPrice"].sum().reset_index().rename(columns={"InvoiceDate": "ds", "TotalPrice": "y"})
...     forecast = Prophet().fit(daily_sales).predict(Prophet().make_future_dataframe(periods=30))
