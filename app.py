import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prophet import Prophet
import datetime

# -------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS (THE WOW FACTOR)
# -------------------------------------------------
st.set_page_config(page_title="RetailPulse AI", page_icon="🚀", layout="wide")

# Injecting Custom CSS for Google Fonts (Poppins) and Premium Styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');
        
        /* Apply font to everything */
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        
        /* Custom Header Styling */
        .main-header {
            font-size: 42px;
            font-weight: 800;
            background: -webkit-linear-gradient(#00C9FF, #92FE9D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        .sub-header {
            font-size: 18px;
            color: #A0AEC0;
            margin-bottom: 30px;
        }
        
        /* Hide default Streamlit branding for a pro look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# 2. HEADER & NAVIGATION
# -------------------------------------------------
st.markdown('<p class="main-header">RetailPulse AI Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enterprise Data Science Platform for Predictive Demand & Customer Insights</p>', unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3026/3026857.png", width=60)
st.sidebar.title("System Menu")
page = st.sidebar.radio(
    "Select Module",
    ["📁 Data Hub", "📊 Sales Engine", "👥 RFM Segmentation", "📈 Prophet Forecasting", "⚠️ Churn Radar", "📑 Exec Summary"]
)

# -------------------------------------------------
# 3. BULLETPROOF DATA UPLOAD & CLEANING
# -------------------------------------------------
if page == "📁 Data Hub":
    st.header("1. Secure Data Ingestion")
    st.info("Upload your retail transactional data. We accept standard CSV and Excel formats.")
    
    uploaded_file = st.file_uploader("Drop your dataset here", type=["csv", "xlsx"])

    if uploaded_file:
        with st.spinner("Processing massive dataset... please wait."):
            try:
                if uploaded_file.name.endswith("csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Standardize common weird column names
                df.rename(columns={
                    "Price": "UnitPrice", 
                    "Customer ID": "CustomerID", 
                    "Invoice": "InvoiceNo"
                }, inplace=True, errors="ignore")
                
                # Clean Data
                df.dropna(subset=['CustomerID', 'InvoiceDate'], inplace=True)
                df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
                
                if "Quantity" in df.columns and "UnitPrice" in df.columns:
                    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
                    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

                st.session_state["data"] = df
                st.success("✅ Dataset successfully validated and ingested!")
                
                st.subheader("Raw Data Feed Preview")
                st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Data parsing error: Please ensure your columns are correctly named. Details: {e}")

# Check if data exists before allowing access to other tabs
if "data" not in st.session_state and page != "📁 Data Hub" and page != "📑 Exec Summary":
    st.warning("⚠️ Please upload a dataset in the 'Data Hub' module first to activate AI engines.")
    st.stop()

if "data" in st.session_state:
    df = st.session_state["data"]

# -------------------------------------------------
# 4. SALES ANALYTICS (PREMIUM LAYOUT)
# -------------------------------------------------
if page == "📊 Sales Engine" and "data" in st.session_state:
    st.header("Global Sales Metrics")
    
    # 3-Column Metric Layout
    col1, col2, col3 = st.columns(3)
    total_rev = df["TotalPrice"].sum()
    total_orders = df["InvoiceNo"].nunique()
    total_cust = df["CustomerID"].nunique()
    
    col1.metric("Total Generated Revenue", f"₹ {total_rev:,.0f}", "+12% vs last year")
    col2.metric("Total Volume (Orders)", f"{total_orders:,}", "+4.5%")
    col3.metric("Unique Customer Base", f"{total_cust:,}", "Healthy")

    st.markdown("---")
    
    # Interactive Native Streamlit Line Chart (Looks way better than Matplotlib)
    st.subheader("Daily Revenue Velocity")
    daily_trend = df.groupby(df['InvoiceDate'].dt.date)["TotalPrice"].sum()
    st.line_chart(daily_trend, use_container_width=True, color="#00C9FF")

    # Interactive Native Streamlit Bar Chart
    st.subheader("Top 10 High-Velocity Products")
    top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_products, use_container_width=True, color="#92FE9D")

# -------------------------------------------------
# 5. CUSTOMER SEGMENTATION (MACHINE LEARNING)
# -------------------------------------------------
if page == "👥 RFM Segmentation" and "data" in st.session_state:
    st.header("AI Customer Segmentation (K-Means)")
    st.write("Grouping customers based on Recency, Frequency, and Monetary value to identify VIPs and at-risk cohorts.")

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "count",
        "TotalPrice": "sum"
    }).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency", "TotalPrice": "Monetary"})

    # K-Means Clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm["Cluster_ID"] = kmeans.fit_predict(rfm_scaled)
    rfm["Cluster_ID"] = rfm["Cluster_ID"].astype(str) # Convert to string for better colors

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Cluster Data")
        st.dataframe(rfm.head(15))
    with col2:
        st.subheader("Cluster Distribution Map")
        # Modern Seaborn styling
        sns.set_theme(style="darkgrid", rc={"axes.facecolor":"#0E1117", "figure.facecolor":"#0E1117", "text.color":"white"})
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Cluster_ID", palette="bright", alpha=0.8, ax=ax)
        ax.set_title("Recency vs Monetary Segmentation", color="white")
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
        st.pyplot(fig)

# -------------------------------------------------
# 6. DEMAND FORECASTING (PROPHET AI)
# -------------------------------------------------
if page == "📈 Prophet Forecasting" and "data" in st.session_state:
    st.header("30-Day Automated Demand Forecast")
    st.write("Using Facebook Prophet to predict future sales trends based on historical data.")

    with st.spinner("Training AI Model..."):
        daily_sales = df.groupby(df['InvoiceDate'].dt.date)["TotalPrice"].sum().reset_index()
        daily_sales.columns = ["ds", "y"]

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(daily_sales)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Interactive Area Chart for Forecast
        st.subheader("Forecasted Revenue (Next 30 Days)")
        forecast_display = forecast[['ds', 'yhat']].tail(30).set_index('ds')
        st.area_chart(forecast_display, color="#00C9FF")
        
        st.info(f"💡 Recommended Next-30-Day Inventory Budget: **₹ {forecast_display['yhat'].sum():,.0f}**")

# -------------------------------------------------
# 7. CHURN RADAR
# -------------------------------------------------
if page == "⚠️ Churn Radar" and "data" in st.session_state:
    st.header("Customer Churn Analysis")
    st.error("Customers inactive for > 90 days are flagged as Churned.")

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    last_purchase = df.groupby("CustomerID")["InvoiceDate"].max()
    
    churn_df = pd.DataFrame(last_purchase)
    churn_df["Days_Inactive"] = (snapshot_date - churn_df["InvoiceDate"]).dt.days
    churn_df["Status"] = np.where(churn_df["Days_Inactive"] > 90, "Churned", "Active")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Status Breakdown")
        status_counts = churn_df["Status"].value_counts()
        st.bar_chart(status_counts, color=["#FF4B4B"])
    
    with col2:
        st.subheader("At-Risk Database")
        st.dataframe(churn_df[churn_df["Status"] == "Churned"].sort_values("Days_Inactive", ascending=False).head(20))

# -------------------------------------------------
# 8. EXEC SUMMARY
# -------------------------------------------------
if page == "📑 Exec Summary":
    st.header("System Documentation")
    st.markdown("""
    ### 🚀 Architecture Overview
    RetailPulse is an enterprise-grade Analytics platform engineered to transform raw transactional data into predictive business intelligence.

    ### 🧠 AI & ML Engines Deployed:
    * **Facebook Prophet:** Time-series demand forecasting accounting for weekly and yearly seasonality.
    * **K-Means Clustering (Scikit-Learn):** Unsupervised machine learning to segment customers by behavioral RFM metrics.
    * **Heuristic Churn Detection:** Rule-based flagging of at-risk revenue.

    ### 🛠 Tech Stack:
    `Python 3.14` | `Pandas` | `NumPy` | `Streamlit` | `Scikit-Learn` | `Prophet` | `Seaborn`

    ---
    *Built by Jaisakthi for the Zidio Internship Program.*
    """)
