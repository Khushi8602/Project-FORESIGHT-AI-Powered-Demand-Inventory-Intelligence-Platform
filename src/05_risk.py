from pathlib import Path
import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


def latest_forecast(data, model, features):
    latest = (data.sort_values("date")
                  .groupby(["store_id", "sku_id"]).tail(1).copy())
    latest["forecast_demand"] = np.clip(model.predict(latest[features]), 0, None)
    return latest


def attach_inventory(latest):
    inv = pd.read_csv(DATA_DIR / "inventory_snapshot.csv")
    inv.columns = inv.columns.str.strip().str.lower()
    inv["last_restock_date"] = pd.to_datetime(inv["last_restock_date"], errors="coerce")
    inv = (inv.sort_values("last_restock_date")
              .drop_duplicates(["store_id", "sku_id"], keep="last"))

    cols = ["store_id", "sku_id", "stock_on_hand", "reorder_point", "safety_stock"]
    latest = latest.merge(inv[cols], on=["store_id", "sku_id"], how="left")
    latest[cols[2:]] = latest[cols[2:]].fillna(0)
    return latest


def compute_risk(latest):
    latest["days_of_cover"] = np.where(
        latest["forecast_demand"] > 0,
        latest["stock_on_hand"] / latest["forecast_demand"],
        999,
    )

    def score(row):
        s = 0
        if row["stock_on_hand"] <= row["reorder_point"]:
            s += 40
        if row["stock_on_hand"] == 0:
            s += 30
        if row["days_of_cover"] < 3:
            s += 20
        elif row["days_of_cover"] < 7:
            s += 10
        if row["forecast_demand"] > row["stock_on_hand"]:
            s += 10
        return min(s, 100)

    latest["risk_score"] = latest.apply(score, axis=1)
    latest["risk_level"] = pd.cut(
        latest["risk_score"], bins=[-1, 39, 69, 100],
        labels=["LOW", "MEDIUM", "HIGH"],
    )
    return latest


def add_rupee_impact(latest):
    sku = pd.read_csv(DATA_DIR / "sku_master.csv")
    sku.columns = sku.columns.str.strip().str.lower()
    latest = latest.merge(sku[["sku_id", "unit_price"]].rename(
        columns={"unit_price": "list_price"}), on="sku_id", how="left")
    latest["list_price"] = latest["list_price"].fillna(latest.get("avg_price", 0))

    unmet = (latest["forecast_demand"] - latest["stock_on_hand"]).clip(lower=0)
    latest["expected_lost_units"] = unmet
    latest["expected_lost_revenue"] = unmet * latest["list_price"]
    return latest


def attach_flags(latest):
    flags = pd.read_csv(DATA_DIR / "sku_inventory_flags.csv")
    flags.columns = flags.columns.str.strip().str.lower()
    flags = flags[["sku_id", "flag"]].drop_duplicates("sku_id")
    return latest.merge(flags, on="sku_id", how="left")


def main():
    data = pd.read_parquet(OUT_DIR / "features.parquet")
    model = joblib.load(MODEL_DIR / "best_model_lgbm.pkl")
    features = joblib.load(MODEL_DIR / "feature_list.pkl")

    latest = latest_forecast(data, model, features)
    latest = attach_inventory(latest)
    latest = compute_risk(latest)
    latest = add_rupee_impact(latest)
    latest = attach_flags(latest)

    out_cols = [
        "store_id", "sku_id", "category", "brand", "forecast_demand",
        "stock_on_hand", "reorder_point", "safety_stock", "days_of_cover",
        "risk_score", "risk_level", "expected_lost_units",
        "expected_lost_revenue", "flag",
    ]
    out_cols = [c for c in out_cols if c in latest.columns]
    latest[out_cols].to_csv(OUT_DIR / "risk_scores.csv", index=False)

    high = (latest["risk_level"] == "HIGH").sum()
    lost = latest["expected_lost_revenue"].sum()
    print(f"[risk] Saved {len(latest):,} rows | HIGH-risk: {high:,} | "
          f"Expected lost revenue: {lost:,.0f}")


if __name__ == "__main__":
    main()
