import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prophet import Prophet

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(page_title="RetailPulse - AI Retail Analytics", layout="wide")

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("📊 RetailPulse - AI Powered Retail Analytics Dashboard")
st.markdown("An end-to-end Data Science and Analytics platform for demand forecasting, customer segmentation, churn analysis, and inventory optimization.")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Go to",
    [
        "📁 Upload Dataset",
        "📊 Sales Analytics",
        "👥 Customer Segmentation",
        "📈 Demand Forecasting",
        "⚠️ Churn Prediction",
        "📦 Inventory Optimization",
        "📑 Project Summary"
    ]
)

# -------------------------------------------------
# DATA UPLOAD
# -------------------------------------------------
if page == "📁 Upload Dataset":
    st.header("Upload Retail Dataset")

    uploaded_file = st.file_uploader("Upload your retail dataset (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state["data"] = df
        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Information")
        st.write(df.describe())

# -------------------------------------------------
# LOAD DATA FROM SESSION
# -------------------------------------------------
if "data" in st.session_state:
    df = st.session_state["data"]

    # Basic preprocessing
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df = df[df["Quantity"] > 0]
        df = df[df["UnitPrice"] > 0]
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# -------------------------------------------------
# SALES ANALYTICS PAGE
# -------------------------------------------------
if page == "📊 Sales Analytics" and "data" in st.session_state:
    st.header("Sales Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    total_revenue = df["TotalPrice"].sum()
    total_orders = df["InvoiceNo"].nunique()
    total_customers = df["CustomerID"].nunique()

    col1.metric("Total Revenue", f"₹ {total_revenue:,.0f}")
    col2.metric("Total Orders", total_orders)
    col3.metric("Total Customers", total_customers)

    st.subheader("Daily Sales Trend")

    daily_sales = df.groupby("InvoiceDate")["TotalPrice"].sum()

    fig = plt.figure()
    daily_sales.plot()
    plt.title("Daily Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    st.pyplot(fig)

    st.subheader("Top 10 Products")

    top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)

    fig2 = plt.figure()
    top_products.plot(kind="bar")
    plt.title("Top 10 Selling Products")
    st.pyplot(fig2)

# -------------------------------------------------
# CUSTOMER SEGMENTATION
# -------------------------------------------------
if page == "👥 Customer Segmentation" and "data" in st.session_state:
    st.header("Customer Segmentation using RFM + KMeans")

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "count",
        "TotalPrice": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    st.subheader("Customer Segmentation Table")
    st.dataframe(rfm.head())

    st.subheader("Cluster Visualization")

    fig3 = plt.figure()
    sns.scatterplot(x="Recency", y="Monetary", hue="Cluster", data=rfm)
    st.pyplot(fig3)

# -------------------------------------------------
# DEMAND FORECASTING
# -------------------------------------------------
if page == "📈 Demand Forecasting" and "data" in st.session_state:
    st.header("Demand Forecasting (30 Days)")

    daily_sales = df.groupby("InvoiceDate")["TotalPrice"].sum().reset_index()
    daily_sales.columns = ["ds", "y"]

    model = Prophet()
    model.fit(daily_sales)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    st.subheader("Forecast Table")
    st.dataframe(forecast[["ds", "yhat"]].tail(30))

    st.subheader("Forecast Plot")

    fig4 = model.plot(forecast)
    st.pyplot(fig4)

# -------------------------------------------------
# CHURN PREDICTION
# -------------------------------------------------
if page == "⚠️ Churn Prediction" and "data" in st.session_state:
    st.header("Customer Churn Prediction")

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    last_purchase = df.groupby("CustomerID")["InvoiceDate"].max()
    churn = (snapshot_date - last_purchase).dt.days > 90

    churn_df = churn.reset_index()
    churn_df.columns = ["CustomerID", "Churn"]

    churn_count = churn_df["Churn"].value_counts()

    fig5 = plt.figure()
    churn_count.plot(kind="bar")
    plt.title("Churn vs Active Customers")
    st.pyplot(fig5)

    st.subheader("Churn Table")
    st.dataframe(churn_df.head())

# -------------------------------------------------
# INVENTORY OPTIMIZATION
# -------------------------------------------------
if page == "📦 Inventory Optimization" and "data" in st.session_state:
    st.header("Inventory Optimization Recommendation")

    daily_sales = df.groupby("InvoiceDate")["TotalPrice"].sum().reset_index()
    daily_sales.columns = ["ds", "y"]

    model = Prophet()
    model.fit(daily_sales)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    recommended_stock = forecast["yhat"].tail(30).sum()

    st.metric("Recommended Stock for Next 30 Days", f"₹ {recommended_stock:,.0f}")

# -------------------------------------------------
# PROJECT SUMMARY
# -------------------------------------------------
if page == "📑 Project Summary":
    st.header("Project Overview")

    st.markdown("""
    ### This dashboard includes:

    ✔ Sales analytics
    ✔ Customer segmentation (RFM + KMeans)
    ✔ Demand forecasting (Prophet Model)
    ✔ Customer churn detection
    ✔ Inventory optimization recommendations

    ### Technologies Used:

    - Python
    - Pandas & NumPy
    - Scikit-learn
    - Prophet
    - Streamlit
    - Data Visualization (Matplotlib, Seaborn)
    """)

    st.success("This is a complete end-to-end data science project ready for portfolio submission.")
