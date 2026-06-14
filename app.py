import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prophet import Prophet

st.set_page_config(page_title="Retail OS", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: rgb(0, 0, 0);
    color: rgb(245, 245, 247);
}

[data-testid="stSidebar"] {
    background-color: rgb(0, 0, 0);
    border-right: 1px solid rgb(51, 51, 51);
}

header {
    visibility: hidden;
}

.glass-card {
    background-color: rgb(28, 28, 30);
    border: 1px solid rgb(51, 51, 51);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    transition: all 0.2s ease;
}

.glass-card:hover {
    border-color: rgb(72, 72, 74);
}

.metric-value {
    font-size: 42px;
    font-weight: 500;
    letter-spacing: -1px;
    color: rgb(255, 255, 255);
}

.metric-label {
    font-size: 13px;
    color: rgb(142, 142, 147);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
    margin-bottom: 8px;
}

.stTextInput input, .stFileUploader {
    background-color: rgb(28, 28, 30) !important;
    border: 1px solid rgb(51, 51, 51) !important;
    color: rgb(255, 255, 255) !important;
    border-radius: 8px !important;
}

.stButton button {
    background-color: rgb(255, 255, 255) !important;
    color: rgb(0, 0, 0) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s;
}

.stButton button:hover {
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
    <style>
    [data-testid="collapsedControl"] { display: none; }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; font-size: 48px; font-weight: 600; letter-spacing: -1.5px;'>Retail OS</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: rgb(142, 142, 147); margin-bottom: 40px;'>Sign in to access enterprise analytics.</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Authenticate", use_container_width=True)
            
            if submit:
                if username == "admin" and password == "admin":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Default is admin/admin")
    st.stop()

st.markdown("""
<style>
[data-testid="collapsedControl"] { display: block; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h3 style='font-weight: 600; letter-spacing: -0.5px; margin-bottom: 20px;'>Navigation</h3>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "",
    ["Executive Overview", "Sales Analytics", "Customer Hub", "Demand Explorer", "Churn Risk", "Inventory Health"]
)

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
if st.sidebar.button("Secure Logout"):
    st.session_state["authenticated"] = False
    st.rerun()

st.markdown("<h1 style='font-size: 44px; font-weight: 600; letter-spacing: -1.5px; margin-bottom: 40px;'>Retail Intelligence OS</h1>", unsafe_allow_html=True)

if page == "Executive Overview":
    st.markdown("<h2 style='font-weight: 500; font-size: 24px;'>System Initialization</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Mount structured dataset payload", type=["csv", "xlsx"])

    if uploaded_file:
        with st.spinner("Authenticating data structures..."):
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
        c1, c2, c3 = st.columns(3)
        
        total_rev = df["TotalPrice"].sum()
        total_orders = df["InvoiceNo"].nunique()
        total_cust = df["CustomerID"].nunique()
        
        c1.markdown(f"<div class='glass-card'><div class='metric-label'>Gross Revenue</div><div class='metric-value'>₹{total_rev:,.0f}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='glass-card'><div class='metric-label'>Total Transactions</div><div class='metric-value'>{total_orders:,}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='glass-card'><div class='metric-label'>Active Cohorts</div><div class='metric-value'>{total_cust:,}</div></div>", unsafe_allow_html=True)

        st.markdown("<h3 style='margin-top: 30px; font-weight: 500; font-size: 20px;'>Revenue Trajectory</h3>", unsafe_allow_html=True)
        daily_trend = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()
        st.line_chart(daily_trend, use_container_width=True)

    if page == "Customer Hub":
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
                sns.set_theme(style="darkgrid", rc={"axes.facecolor":"rgb(28,28,30)", "figure.facecolor":"rgb(0,0,0)", "text.color":"rgb(245,245,247)", "grid.color": "rgb(51,51,51)"})
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Cluster", palette="light:w_r", alpha=0.9, ax=ax, legend=False)
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.tick_params(colors="white")
                st.pyplot(fig)
        except Exception as e:
            st.error("Insufficient variance to process clustering.")

    if page == "Demand Explorer":
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
                st.error("Time-series continuity broken.")

    if page == "Churn Risk":
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
                st.info("Logistics model advises securing capital matching the projected demand.")
            except Exception as e:
                st.error("Logistics model offline.")
