import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prophet import Prophet

st.set_page_config(page_title="RetailPulse OS", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: rgb(10, 10, 12);
    color: rgb(240, 240, 240);
}

[data-testid="stSidebar"] {
    background-color: rgba(20, 20, 25, 0.8);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.5px;
}

.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.metric-value {
    font-size: 42px;
    font-weight: 600;
    background: linear-gradient(90deg, rgb(0, 122, 255), rgb(48, 209, 88));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 52px; font-weight: 700; letter-spacing: -1.5px; margin-bottom: 0;'>RetailPulse OS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: rgb(160, 160, 170); font-size: 18px; margin-bottom: 50px;'>Intelligent Commerce Analytics</p>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "System Navigation",
    ["Data Hub", "Sales Intelligence", "Customer Clusters", "Demand Forecast", "Attrition Radar"]
)

if page == "Data Hub":
    st.markdown("<h2>Data Ingestion</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Select structured dataset", type=["csv", "xlsx"])

    if uploaded_file:
        with st.spinner("Synchronizing data..."):
            if uploaded_file.name.endswith("csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            df.rename(columns={
                "Price": "UnitPrice", 
                "Customer ID": "CustomerID", 
                "Invoice": "InvoiceNo"
            }, inplace=True, errors="ignore")
            
            df.dropna(subset=["CustomerID", "InvoiceDate"], inplace=True)
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
            
            if "Quantity" in df.columns and "UnitPrice" in df.columns:
                df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
                df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

            st.session_state["data"] = df
            st.success("System aligned. Data mapped successfully.")
            st.dataframe(df.head(15), use_container_width=True)

if "data" not in st.session_state and page != "Data Hub":
    st.warning("Awaiting data payload in Data Hub.")
    st.stop()

if "data" in st.session_state:
    df = st.session_state["data"]

    if page == "Sales Intelligence":
        st.markdown("<h2>Performance Metrics</h2>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        
        total_rev = df["TotalPrice"].sum()
        total_orders = df["InvoiceNo"].nunique()
        total_cust = df["CustomerID"].nunique()
        
        c1.markdown(f"<div class='glass-card'><h3>Gross Volume</h3><div class='metric-value'>₹{total_rev:,.0f}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='glass-card'><h3>Transactions</h3><div class='metric-value'>{total_orders:,}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='glass-card'><h3>Active Entities</h3><div class='metric-value'>{total_cust:,}</div></div>", unsafe_allow_html=True)

        st.markdown("<h3>Velocity Tracker</h3>", unsafe_allow_html=True)
        daily_trend = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()
        st.line_chart(daily_trend, use_container_width=True)

    if page == "Customer Clusters":
        st.markdown("<h2>Algorithmic Segmentation</h2>", unsafe_allow_html=True)
        
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        rfm = df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "InvoiceNo": "count",
            "TotalPrice": "sum"
        }).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency", "TotalPrice": "Monetary"})

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)
        kmeans = KMeans(n_clusters=4, random_state=42)
        rfm["Cluster"] = kmeans.fit_predict(rfm_scaled).astype(str)

        c1, c2 = st.columns([1, 2])
        c1.dataframe(rfm.head(15))
        
        with c2:
            sns.set_theme(style="darkgrid", rc={"axes.facecolor":"rgb(20,20,25)", "figure.facecolor":"rgb(10,10,12)", "text.color":"rgb(200,200,200)"})
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Cluster", palette="viridis", alpha=0.9, ax=ax)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.tick_params(colors="white")
            st.pyplot(fig)

    if page == "Demand Forecast":
        st.markdown("<h2>Predictive Engine</h2>", unsafe_allow_html=True)
        
        with st.spinner("Computing trajectories..."):
            daily_sales = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum().reset_index()
            daily_sales.columns = ["ds", "y"]

            model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            model.fit(daily_sales)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            forecast_display = forecast[["ds", "yhat"]].tail(30).set_index("ds")
            st.area_chart(forecast_display)
            
            projected_value = forecast_display["yhat"].sum()
            st.markdown(f"<div class='glass-card'><h3>30-Day Automated Outlook</h3><div class='metric-value'>₹{projected_value:,.0f}</div></div>", unsafe_allow_html=True)

    if page == "Attrition Radar":
        st.markdown("<h2>Risk Analysis</h2>", unsafe_allow_html=True)
        
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        last_purchase = df.groupby("CustomerID")["InvoiceDate"].max()
        
        churn_df = pd.DataFrame(last_purchase)
        churn_df["Days_Inactive"] = (snapshot_date - churn_df["InvoiceDate"]).dt.days
        churn_df["Status"] = np.where(churn_df["Days_Inactive"] > 90, "At Risk", "Secure")
        
        c1, c2 = st.columns(2)
        status_counts = churn_df["Status"].value_counts()
        c1.bar_chart(status_counts)
        c2.dataframe(churn_df[churn_df["Status"] == "At Risk"].sort_values("Days_Inactive", ascending=False).head(20))
