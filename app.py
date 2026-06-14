import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prophet import Prophet

st.set_page_config(page_title="Retail OS", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: rgb(0, 0, 0);
    color: rgb(245, 245, 247);
}

[data-testid="stSidebar"] {
    background-color: rgba(28, 28, 30, 0.5);
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

.glass-card {
    background: rgba(44, 44, 46, 0.4);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    transition: all 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-3px);
    background: rgba(58, 58, 60, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.metric-value {
    font-size: 46px;
    font-weight: 500;
    letter-spacing: -1px;
    background: linear-gradient(135deg, rgb(10, 132, 255), rgb(94, 92, 230));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 14px;
    color: rgb(142, 142, 147);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 56px; font-weight: 600; letter-spacing: -2px; margin-bottom: 5px; background: linear-gradient(90deg, rgb(255,255,255), rgb(150,150,150)); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Retail Intelligence OS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: rgb(142, 142, 147); font-size: 20px; font-weight: 300; margin-bottom: 60px;'>Enterprise Analytics Environment</p>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Executive Overview", "Sales Analytics", "Customer Hub", "Demand Explorer", "Churn Risk", "Inventory Health"]
)

if page == "Executive Overview":
    st.markdown("<h2 style='font-weight: 500; letter-spacing: -0.5px;'>System Initialization</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Mount structured dataset payload", type=["csv", "xlsx"])

    if uploaded_file:
        with st.spinner("Authenticating and parsing data structures..."):
            try:
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
                st.success("Environment secured. Data matrix loaded.")
                st.dataframe(df.head(15), use_container_width=True)
            except Exception as e:
                st.error("Matrix alignment failed. Verify dataset integrity.")

if "data" not in st.session_state and page != "Executive Overview":
    st.warning("Awaiting data payload in Executive Overview.")
    st.stop()

if "data" in st.session_state:
    df = st.session_state["data"]

    if page == "Sales Analytics":
        st.markdown("<h2 style='font-weight: 500; letter-spacing: -0.5px;'>Sales Engine Diagnostics</h2>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        
        total_rev = df["TotalPrice"].sum()
        total_orders = df["InvoiceNo"].nunique()
        total_cust = df["CustomerID"].nunique()
        
        c1.markdown(f"<div class='glass-card'><div class='metric-label'>Gross Revenue</div><div class='metric-value'>₹{total_rev:,.0f}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='glass-card'><div class='metric-label'>Total Transactions</div><div class='metric-value'>{total_orders:,}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='glass-card'><div class='metric-label'>Active Cohorts</div><div class='metric-value'>{total_cust:,}</div></div>", unsafe_allow_html=True)

        st.markdown("<h3 style='margin-top: 30px; font-weight: 500;'>Revenue Trajectory</h3>", unsafe_allow_html=True)
        daily_trend = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()
        st.line_chart(daily_trend, use_container_width=True)

    if page == "Customer Hub":
        st.markdown("<h2 style='font-weight: 500; letter-spacing: -0.5px;'>Algorithmic Customer Clustering</h2>", unsafe_allow_html=True)
        
        try:
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
                sns.set_theme(style="darkgrid", rc={"axes.facecolor":"rgb(28,28,30)", "figure.facecolor":"rgb(0,0,0)", "text.color":"rgb(245,245,247)"})
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Cluster", palette="cool", alpha=0.9, ax=ax)
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.tick_params(colors="white")
                st.pyplot(fig)
        except Exception as e:
            st.error("Insufficient variance to process clustering. Require larger dataset.")

    if page == "Demand Explorer":
        st.markdown("<h2 style='font-weight: 500; letter-spacing: -0.5px;'>Predictive Demand Models</h2>", unsafe_allow_html=True)
        
        with st.spinner("Calibrating time-series models..."):
            try:
                daily_sales = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum().reset_index()
                daily_sales.columns = ["ds", "y"]

                model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                model.fit(daily_sales)
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)

                forecast_display = forecast[["ds", "yhat"]].tail(30).set_index("ds")
                st.area_chart(forecast_display)
            except Exception as e:
                st.error("Time-series continuity broken. Verify timestamp integrity.")

    if page == "Churn Risk":
        st.markdown("<h2 style='font-weight: 500; letter-spacing: -0.5px;'>Attrition Identification Radar</h2>", unsafe_allow_html=True)
        
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        last_purchase = df.groupby("CustomerID")["InvoiceDate"].max()
        
        churn_df = pd.DataFrame(last_purchase)
        churn_df["Days_Inactive"] = (snapshot_date - churn_df["InvoiceDate"]).dt.days
        churn_df["Status"] = np.where(churn_df["Days_Inactive"] > 90, "At Risk", "Secure")
        
        c1, c2 = st.columns(2)
        status_counts = churn_df["Status"].value_counts()
        c1.bar_chart(status_counts)
        c2.dataframe(churn_df[churn_df["Status"] == "At Risk"].sort_values("Days_Inactive", ascending=False).head(20))

    if page == "Inventory Health":
        st.markdown("<h2 style='font-weight: 500; letter-spacing: -0.5px;'>Automated Inventory Logistics</h2>", unsafe_allow_html=True)
        
        with st.spinner("Calculating optimal capital deployment..."):
            try:
                daily_sales = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum().reset_index()
                daily_sales.columns = ["ds", "y"]

                model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                model.fit(daily_sales)
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)

                projected_value = forecast["yhat"].tail(30).sum()
                
                st.markdown(f"<div class='glass-card'><div class='metric-label'>Recommended 30-Day Restock Capital</div><div class='metric-value'>₹{projected_value:,.0f}</div></div>", unsafe_allow_html=True)
                
                st.info("Logistics model advises securing capital matching the projected demand to prevent out-of-stock scenarios.")
            except Exception as e:
                st.error("Logistics model offline. Require stable demand history.")
