from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"


def build_daily(df):
    daily = (
        df.groupby(["date", "store_id", "sku_id"])
          .agg(quantity=("quantity", "sum"),
               revenue=("net_value", "sum"),
               avg_price=("unit_price", "mean"),
               had_promo=("has_promo", "max"))
          .reset_index()
    )
    return daily


def add_calendar(daily):
    d = daily["date"].dt
    daily["year"] = d.year
    daily["month"] = d.month
    daily["day"] = d.day
    daily["dayofweek"] = d.dayofweek
    daily["weekofyear"] = d.isocalendar().week.astype(int)
    daily["is_weekend"] = (daily["dayofweek"] >= 5).astype(int)
    daily["is_month_start"] = d.is_month_start.astype(int)
    daily["is_month_end"] = d.is_month_end.astype(int)
    return daily


def add_lags(daily):
    daily = daily.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    grp = daily.groupby(["store_id", "sku_id"])["quantity"]

    for lag in (1, 7, 14, 28):
        daily[f"lag_{lag}"] = grp.shift(lag)

    shifted = grp.shift(1)
    for w in (7, 14, 28):
        daily[f"roll_mean_{w}"] = shifted.rolling(w).mean()
    daily["roll_std_7"] = shifted.rolling(7).std()

    lag_cols = [c for c in daily.columns if c.startswith(("lag_", "roll_"))]
    daily[lag_cols] = daily[lag_cols].fillna(0)
    return daily


def add_categoricals(daily, df):
    meta = df[["sku_id", "category", "brand"]].drop_duplicates("sku_id")
    daily = daily.merge(meta, on="sku_id", how="left")
    daily["category"] = daily["category"].fillna("Unknown")
    daily["brand"] = daily["brand"].fillna("Unknown")

    for col in ("category", "brand", "store_id"):
        daily[col + "_code"] = daily[col].astype("category").cat.codes
    return daily


def main():
    df = pd.read_parquet(OUT_DIR / "sales_clean.parquet")
    daily = build_daily(df)
    daily = add_calendar(daily)
    daily = add_lags(daily)
    daily = add_categoricals(daily, df)

    daily.to_parquet(OUT_DIR / "features.parquet", index=False)
    print(f"[features] Saved {len(daily):,} rows -> outputs/features.parquet")


if __name__ == "__main__":
    main()
