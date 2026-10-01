from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="FORESIGHT – AI-Powered Demand & Inventory Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>.main {padding-top: 1rem;}
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 12px;
        border-radius: 12px;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,unsafe_allow_html=True,
)

def money(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "₹0"
    if abs(value) >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    if abs(value) >= 1e5:
        return f"₹{value / 1e5:.2f} L"
    if abs(value) >= 1e3:
        return f"₹{value / 1e3:.1f} K"
    return f"₹{value:,.0f}"


def first_existing(df, names, default=None):
    for name in names:
        if name in df.columns:
            return name
    return default


def numeric_sum(df, column):
    return float(df[column].sum()) if column and column in df.columns else 0.0


def safe_div(a, b):
    return a / b if b not in (0, None) else 0


def standardize_columns(df):
    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    ]
    return df


@st.cache_data
def load_csv(path):
    if path.exists():
        try:
            return standardize_columns(pd.read_csv(path))
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data
def load_project_data():
    possible_sales = [
        OUT_DIR / "sales_clean.csv",
        DATA_DIR / "processed" / "sales_clean.csv",
        DATA_DIR / "sales_clean.csv",
        DATA_DIR / "raw" / "sales_raw.csv",
    ]
    possible_risk = [
        OUT_DIR / "risk_scores.csv",
        DATA_DIR / "processed" / "risk_scores.csv",
        DATA_DIR / "risk_scores.csv",
    ]
    possible_forecast = [
        OUT_DIR / "forecast.csv",
        DATA_DIR / "processed" / "forecast.csv",
        DATA_DIR / "forecast.csv",
    ]

    sales = pd.DataFrame()
    risk = pd.DataFrame()
    forecast = pd.DataFrame()

    for path in possible_sales:
        sales = load_csv(path)
        if not sales.empty:
            break

    for path in possible_risk:
        risk = load_csv(path)
        if not risk.empty:
            break

    for path in possible_forecast:
        forecast = load_csv(path)
        if not forecast.empty:
            break

    if "date" in sales.columns:
        sales["date"] = pd.to_datetime(sales["date"], errors="coerce")

    if "date" in forecast.columns:
        forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")

    return sales, risk, forecast


sales_df, risk_df, forecast_df = load_project_data()


if not sales_df.empty:
    if "net_value" not in sales_df.columns:
        if {"units", "price"}.issubset(sales_df.columns):
            sales_df["net_value"] = sales_df["units"] * sales_df["price"]
        elif {"quantity", "unit_price"}.issubset(sales_df.columns):
            sales_df["net_value"] = sales_df["quantity"] * sales_df["unit_price"]

    if "profit" not in sales_df.columns and {"net_value", "cost"}.issubset(sales_df.columns):
        sales_df["profit"] = sales_df["net_value"] - sales_df["cost"]

    if "units" not in sales_df.columns and "quantity" in sales_df.columns:
        sales_df["units"] = sales_df["quantity"]

if not risk_df.empty:
    if "risk_level" not in risk_df.columns and "risk_score" in risk_df.columns:
        risk_df["risk_level"] = pd.cut(
            risk_df["risk_score"],
            bins=[-np.inf, 33, 66, np.inf],
            labels=["Low", "Medium", "High"],
        ).astype(str)

    if "expected_lost_revenue" not in risk_df.columns:
        if {"demand", "on_hand", "unit_price"}.issubset(risk_df.columns):
            shortage = (risk_df["demand"] - risk_df["on_hand"]).clip(lower=0)
            risk_df["expected_lost_revenue"] = shortage * risk_df["unit_price"]
        else:
            risk_df["expected_lost_revenue"] = 0


st.sidebar.title("📊 Project Foresight")
st.sidebar.caption("Demand & Inventory Intelligence Platform")

page = st.sidebar.radio("Navigate",["🏠 Home","📈 Sales Analytics","🔮 Demand Forecast","📦 Inventory Dashboard","⚠️ Risk Dashboard","🔍 Product Details","📊 Executive Summary Dashboard",],)

st.sidebar.markdown("---")
st.sidebar.subheader("Project Information")
st.sidebar.write("**Domain:** Retail / Supply Chain")
st.sidebar.write("**Core:** Demand Forecasting + Risk Scoring")
st.sidebar.write("**Framework:** Streamlit + Plotly")

if sales_df.empty:
    st.sidebar.error("Sales data not found.")
if risk_df.empty:
    st.sidebar.warning("Risk data not found. Risk pages will show limited information.")


revenue_col = first_existing(sales_df, ["net_value", "revenue", "sales"])
profit_col = first_existing(sales_df, ["profit", "net_profit"])
units_col = first_existing(sales_df, ["units", "quantity"])
sku_col = first_existing(sales_df, ["sku_id", "sku", "product_id"])

total_revenue = numeric_sum(sales_df, revenue_col)
total_profit = numeric_sum(sales_df, profit_col)
total_units = numeric_sum(sales_df, units_col)
margin = safe_div(total_profit * 100, total_revenue)

risk_sku_col = first_existing(risk_df, ["sku_id", "sku", "product_id"])
risk_count = risk_df[risk_sku_col].nunique() if risk_sku_col else len(risk_df)
high_risk_count = (
    int((risk_df["risk_level"].astype(str).str.lower() == "high").sum())
    if "risk_level" in risk_df.columns else 0
)
lost_revenue = numeric_sum(risk_df, "expected_lost_revenue")


def show_kpis():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", money(total_revenue))
    c2.metric("Total Profit", money(total_profit))
    c3.metric("Units Sold", f"{total_units:,.0f}")
    c4.metric("Profit Margin", f"{margin:.1f}%")


def revenue_trend():
    if {"date", revenue_col}.issubset(sales_df.columns):
        trend = (sales_df.dropna(subset=["date"]).groupby("date")[revenue_col].sum().reset_index())
        if not trend.empty:
            fig = px.line(trend,x="date",y=revenue_col,markers=True,title="Revenue Trend",labels={revenue_col: "Revenue (₹)", "date": "Date"},)
            st.plotly_chart(fig, use_container_width=True)


def demand_series():
    if {"date", units_col}.issubset(sales_df.columns):
        result = (sales_df.dropna(subset=["date"]).groupby("date")[units_col].sum().reset_index().sort_values("date"))
        result["moving_average_7"] = result[units_col].rolling(7, min_periods=1).mean()
        return result
    return pd.DataFrame()

if page == "🏠 Home":
    st.title("📊 Project Foresight")
    st.subheader("AI-Powered Demand & Inventory Intelligence Platform")

    st.markdown(
        """
        **Project Objective:** To analyze retail sales data to provide demand forecasting, inventory monitoring and inventory risk identification.

        ### Project Modules
        - Sales Analytics
        - Demand Forecast
        - Inventory  Dashboard
        - Risk Dashboard
        - Product Details 
        - Executive Summary Dashboard
        """
    )

    st.markdown("### Quick Overview")
    show_kpis()

    st.markdown("### Technology Stack")
    cols = st.columns(5)
    for col, item in zip(cols, ["Python", "Pandas", "Scikit-learn", "Plotly", "Streamlit"]):
        col.info(item)

    st.markdown("### Project Architecture")
    st.code(
        """
Raw Data
   ↓
Data Cleaning
   ↓
EDA + Feature Engineering
   ↓
Forecasting Model + Risk Scoring Engine
   ↓
Streamlit Dashboard
   ↓
Deployment / Prediction API
        """,
        language="text",)

    if not sales_df.empty:
        st.success(f"Sales dataset loaded successfully: {len(sales_df):,} records")
    else:
        st.warning("Sales dataset not loading. Keep required characters in outputs/sales_clean.csv.")


elif page == "📈 Sales Analytics":
    st.title("📈 Sales Analytics")
    show_kpis()

    if sales_df.empty:
        st.warning("Sales data is not available.")
    else:
        revenue_trend()

        dimensions = [col for col in ["category", "brand", "store_id", "channel", "region"] if col in sales_df.columns]

        if dimensions and revenue_col:
            st.markdown("### Revenue Breakdown")
            dimension = st.selectbox("Breakdown by", dimensions)
            grouped = (sales_df.groupby(dimension)[revenue_col].sum().sort_values(ascending=False).head(15).reset_index())
            fig = px.bar(grouped,x=dimension,y=revenue_col,title=f"Revenue by {dimension}",labels={revenue_col: "Revenue (₹)"},)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(grouped, use_container_width=True)

        if {"date", revenue_col}.issubset(sales_df.columns):
            st.markdown("### Revenue by Weekday")
            weekday_df = sales_df.dropna(subset=["date"]).copy()
            weekday_df["weekday"] = weekday_df["date"].dt.day_name()
            order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
            weekday = (weekday_df.groupby("weekday")[revenue_col].sum().reindex(order).reset_index())
            fig = px.bar(weekday,x="weekday",y=revenue_col,title="Revenue by Weekday",labels={revenue_col: "Revenue (₹)"},)
            st.plotly_chart(fig, use_container_width=True)

        if sku_col and revenue_col:
            st.markdown("### Top Products / SKUs")
            top_skus = (sales_df.groupby(sku_col)[revenue_col].sum().sort_values(  ascending=False).head(10).reset_index())
            fig = px.bar(top_skus,x=sku_col,y=revenue_col,title="Top 10 SKUs by Revenue",labels={revenue_col: "Revenue (₹)"},)
            st.plotly_chart(fig, use_container_width=True)


elif page == "🔮 Demand Forecast":
    st.title("🔮 Demand Forecast")
    st.caption("The forecast page displays historical demand and the available forecast output.")

    if not sales_df.empty and units_col:
        series = demand_series()

        if not series.empty:
            st.markdown("### Historical Demand")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series["date"],y=series[units_col],mode="lines",name="Actual Units",))
            fig.add_trace(go.Scatter(x=series["date"],y=series["moving_average_7"],mode="lines", name="7-Day Moving Average",))
            fig.update_layout(title="Actual Demand vs 7-Day Moving Average",xaxis_title="Date",yaxis_title="Units",)
            st.plotly_chart(fig, use_container_width=True)

            st.metric("Next Period Baseline Estimate",f"{series['moving_average_7'].iloc[-1]:,.0f} units",)

        if not forecast_df.empty:
            st.markdown("### Model Forecast Output")
            st.dataframe(forecast_df, use_container_width=True)

            forecast_date = first_existing(forecast_df, ["date", "forecast_date"])
            forecast_value = first_existing(
                forecast_df, ["forecast", "predicted_demand", "prediction", "yhat"]
            )
            if forecast_date and forecast_value:
                fig = px.line(forecast_df,x=forecast_date,y=forecast_value,markers=True,title="Forecasted Demand",labels={forecast_value: "Predicted Demand"},
                )
                st.plotly_chart(fig, use_container_width=True)

        metrics_path = MODEL_DIR / "model_metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                st.markdown("### Model Performance")
                cols = st.columns(len(metrics))
                for col, (key, value) in zip(cols, metrics.items()):
                    col.metric(str(key).upper(), str(value))
            except Exception:
                st.info("The model metrics file couldn't be read.")
    else:
        st.warning("Date and units/quantity columns are required for the forecast.")


elif page == "📦 Inventory Dashboard":
    st.title("📦 Inventory Dashboard")

    if risk_df.empty:
        st.warning("The risk_scores.csv for inventory data is not available.")
    else:
        on_hand_col = first_existing(risk_df, ["on_hand", "stock", "current_stock"])
        reorder_col = first_existing(risk_df, ["reorder_point", "reorder_level"])
        status_col = first_existing(risk_df, ["inventory_status", "stock_status"])

        low_stock = 0
        out_of_stock = 0
        reorder_required = 0

        if on_hand_col:
            out_of_stock = int((risk_df[on_hand_col].fillna(0) <= 0).sum())
        if on_hand_col and reorder_col:
            reorder_required = int(
                (risk_df[on_hand_col].fillna(0) <= risk_df[reorder_col].fillna(0)).sum()
            )
            low_stock = reorder_required - out_of_stock

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total SKUs", f"{risk_count:,}")
        c2.metric("Low Stock", f"{max(low_stock, 0):,}")
        c3.metric("Out of Stock", f"{out_of_stock:,}")
        c4.metric("Reorder Required", f"{reorder_required:,}")

        st.metric("Expected Lost Revenue", money(lost_revenue))

        display_cols = [
            col for col in ["sku_id", "category", "brand", on_hand_col, reorder_col,"demand", "risk_level", "risk_score", "expected_lost_revenue"] if col and col in risk_df.columns]

        if display_cols:
            st.dataframe(risk_df[display_cols], use_container_width=True)
        else:
            st.dataframe(risk_df, use_container_width=True)

        st.download_button(
            "⬇️ Download Inventory Report",
            data=risk_df.to_csv(index=False).encode("utf-8"),
            file_name="inventory_report.csv",
            mime="text/csv",
        )

elif page == "⚠️ Risk Dashboard":
    st.title("⚠️ Risk Dashboard")

    if risk_df.empty:
        st.warning("Risk data is not available.")
    else:
        if "risk_level" in risk_df.columns:
            risk_counts = (risk_df["risk_level"].astype(str).value_counts().reset_index())
            risk_counts.columns = ["risk_level", "count"]

            fig = px.bar(
                risk_counts,
                x="risk_level",
                y="count",
                title="Risk Level Distribution",
                labels={"risk_level": "Risk Level", "count": "Number of SKUs"},
            )
            st.plotly_chart(fig, use_container_width=True)

            levels = sorted(risk_df["risk_level"].dropna().astype(str).unique().tolist())
            selected = st.multiselect(
                "Filter Risk Level",
                levels,
                default=levels,
            )
            filtered = risk_df[risk_df["risk_level"].astype(str).isin(selected)]
        else:
            filtered = risk_df.copy()
            st.info("risk_level column is  not available.")

        if "expected_lost_revenue" in filtered.columns:
            filtered = filtered.sort_values(
                "expected_lost_revenue", ascending=False
            )

        st.markdown("### Risk Priority Table")
        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            "⬇️ Download Risk / Reorder List",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="risk_reorder_list.csv",
            mime="text/csv",
        )


elif page == "🔍 Product Details":
    st.title("🔍 Product Details")

    if sales_df.empty or not sku_col:
        st.warning("Sales data needs the sku_id/product_id for product details.")
    else:
        sku_values = sorted(sales_df[sku_col].dropna().astype(str).unique().tolist())
        selected_sku = st.selectbox("Select SKU / Product", sku_values)
        product_data = sales_df[sales_df[sku_col].astype(str) == selected_sku]

        product_revenue = numeric_sum(product_data, revenue_col)
        product_units = numeric_sum(product_data, units_col)

        c1, c2, c3 = st.columns(3)
        c1.metric("Product Revenue", money(product_revenue))
        c2.metric("Units Sold", f"{product_units:,.0f}")
        c3.metric("Records", f"{len(product_data):,}")

        if {"date", units_col}.issubset(product_data.columns):
            trend = (product_data.dropna(subset=["date"]).groupby("date")[units_col].sum().reset_index())
            fig = px.line(trend,x="date",y=units_col,markers=True,title=f"Demand Trend — {selected_sku}",labels={units_col: "Units"},)
            st.plotly_chart(fig, use_container_width=True)

        if risk_sku_col:
            product_risk = risk_df[
                risk_df[risk_sku_col].astype(str) == selected_sku
            ]
            if not product_risk.empty:
                st.markdown("### Product Risk Snapshot")
                st.dataframe(product_risk, use_container_width=True)
            else:
                st.info("Risk record is not available for this SKU.")

elif page == "📊 Executive Summary Dashboard":
    st.title("📊 Executive Summary Dashboard")
    st.caption("Management-level view of sales, demand, inventory and risk.")

    show_kpis()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracked SKUs", f"{risk_count:,}")
    c2.metric("High-Risk SKUs", f"{high_risk_count:,}")
    c3.metric("Expected Lost Revenue", money(lost_revenue))
    c4.metric("Forecast Data Rows", f"{len(forecast_df):,}")

    st.markdown("### Revenue Performance")
    revenue_trend()

    if not risk_df.empty and "risk_level" in risk_df.columns:
        st.markdown("### Risk Overview")
        risk_summary = (
            risk_df["risk_level"].astype(str)
            .value_counts()
            .reset_index()
        )
        risk_summary.columns = ["risk_level", "count"]
        fig = px.pie(
            risk_summary,
            names="risk_level",
            values="count",
            title="SKU Risk Composition",
            hole=0.45,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Business Recommendations")
    recommendations = [
        "Define reorder priority for high-risk SKUs.",
        "Integrate demand forecast with inventory planning.",
        "Maintain safety stock for low-stock products.",
        "Make inventory decisions based on revenue and lost-revenue impact.",
        "Monitor the forecast model using MAE, RMSE, and MAPE.",
    ]
    for item in recommendations:
        st.write(f"• {item}")

st.markdown("---")
st.caption("Project Foresight | AI-Demand Forecasting & Inventory Intelligence")
