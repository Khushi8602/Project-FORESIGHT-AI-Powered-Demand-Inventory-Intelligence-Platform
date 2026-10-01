from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"

sns.set_style("whitegrid")


def load():
    return pd.read_parquet(OUT_DIR / "sales_clean.parquet")


def plot_daily_trend(df):
    daily = df.groupby("date")["net_value"].sum().reset_index()
    plt.figure(figsize=(14, 5))
    plt.plot(daily["date"], daily["net_value"], linewidth=0.8, color="#1E3A8A")
    plt.title("Daily total revenue for the entire time period")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "daily_sales_trend.png", dpi=120)
    plt.close()


def plot_top10(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    df.groupby("category")["net_value"].sum().nlargest(10).plot(
        kind="barh", ax=axes[0], color="teal")
    axes[0].set_title("Top 10 Categories (revenue)")
    axes[0].invert_yaxis()

    df.groupby("brand")["net_value"].sum().nlargest(10).plot(
        kind="barh", ax=axes[1], color="purple")
    axes[1].set_title("Top 10 Brands (revenue)")
    axes[1].invert_yaxis()

    df.groupby("store_id")["net_value"].sum().nlargest(10).plot(
        kind="barh", ax=axes[2], color="darkgreen")
    axes[2].set_title("Top 10 Stores (revenue)")
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "top10.png", dpi=120)
    plt.close()


def plot_channel_and_weekday(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df.groupby("channel")["net_value"].sum().sort_values().plot(
        kind="barh", ax=axes[0], color="#0EA5E9")
    axes[0].set_title("Channel-wise revenue")

    wd = df.copy()
    wd["weekday"] = wd["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
    wd.groupby("weekday")["net_value"].sum().reindex(order).plot(
        kind="bar", ax=axes[1], color="#F59E0B")
    axes[1].set_title("Weekday-wise revenue")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "channel_weekday.png", dpi=120)
    plt.close()


def print_summary(df):
    print("\n===== EDA SUMMARY =====")
    print(f"Rows : {len(df):,}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Unique SKUs : {df['sku_id'].nunique():,}")
    print(f"Unique stores : {df['store_id'].nunique()}")
    print(f"Unique customers: {df['customer_id'].nunique():,}")
    print(f"Total revenue : {df['net_value'].sum():,.0f}")
    print(f"Total profit : {df['profit'].sum():,.0f}")
    print(f"Promo share : {df['has_promo'].mean() * 100:.1f}% line items")


def main():
    df = load()
    plot_daily_trend(df)
    plot_top10(df)
    plot_channel_and_weekday(df)
    print_summary(df)
    print("[eda] Charts saved to outputs/")


if __name__ == "__main__":
    main()
