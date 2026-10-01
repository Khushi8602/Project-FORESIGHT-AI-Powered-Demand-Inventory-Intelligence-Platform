from pathlib import Path
import pandas as pd
import numpy as np




BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_raw():
    sales = pd.read_csv(DATA_DIR / "sales_daily.csv")
    sku = pd.read_csv(DATA_DIR / "sku_master.csv")
    stores = pd.read_csv(DATA_DIR / "store_master.csv")

    sales = sales.loc[:, ~sales.columns.str.startswith("Unnamed")]
    for df in (sales, sku, stores):
        df.columns = df.columns.str.strip().str.lower()

    return sales, sku, stores


def clean_sales(sales, sku, stores):
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    sales = sales.dropna(subset=["date"])


    before = len(sales)
    sales = sales.drop_duplicates()
    removed = before - len(sales)
    print(f"[clean] {removed:,} duplicate line items have been removed")

    sales = sales[(sales["quantity"] > 0) & (sales["unit_price"] > 0)]

    sales["promo_id"] = sales["promo_id"].fillna("NO_PROMO").replace("", "NO_PROMO")
    sales["has_promo"] = (sales["promo_id"] != "NO_PROMO").astype(int)
    sales["discount_pct"] = sales["discount_pct"].fillna(0.0)

    fallback = sales["quantity"] * sales["unit_price"] * (1 - sales["discount_pct"] / 100.0)
    sales["net_value"] = sales["total_value"].where(sales["total_value"].notna(), fallback)

    sku_cols = ["sku_id", "category", "brand", "cost_price"]
    sales = sales.merge(sku[sku_cols], on="sku_id", how="left")

    store_cols = ["store_id", "city", "store_type"]
    sales = sales.merge(stores[store_cols], on="store_id", how="left")

    sales["cost_total"] = sales["quantity"] * sales["cost_price"]
    sales["profit"] = sales["net_value"] - sales["cost_total"]

    sales["category"] = sales["category"].fillna("Unknown")
    sales["brand"] = sales["brand"].fillna("Unknown")
    sales["city"] = sales["city"].fillna("Unknown")
    return sales

def main():
    sales, sku, stores = load_raw()
    clean = clean_sales(sales, sku, stores)
    keep = [
        "date", "receipt_id", "store_id", "sku_id", "customer_id",
        "quantity", "unit_price", "net_value", "profit", "cost_total",
        "channel", "discount_pct", "has_promo", "promo_id",
        "category", "brand", "city", "store_type",
    ]
    keep = [c for c in keep if c in clean.columns]
    clean = clean[keep]

    clean.to_parquet(OUTPUT_DIR / "sales_clean.parquet", index=False)
    clean.to_csv(OUTPUT_DIR / "sales_clean.csv", index=False)

    print(f"[clean] Saved {len(clean):,} rows -> outputs/sales_clean.parquet")
    print(f"[clean] Date range: {clean['date'].min().date()} to {clean['date'].max().date()}")
    print(f"[clean] Total revenue: {clean['net_value'].sum():,.0f} | "
          f"Total profit: {clean['profit'].sum():,.0f}")


if __name__ == "__main__":
    main()
