from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "year", "month", "day", "dayofweek", "weekofyear", "is_weekend",
    "is_month_start", "is_month_end", "had_promo", "avg_price",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_std_7",
    "category_code", "brand_code", "store_id_code",
]
TARGET = "quantity"


def wape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    return np.sum(np.abs(y_true - y_pred)) / denom if denom else np.nan


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    w = wape(y_true, y_pred)
    print(f"  {name:16s} | MAE={mae:6.3f} | RMSE={rmse:6.3f} | WAPE={w*100:5.1f}%")
    return {"model": name, "MAE": mae, "RMSE": rmse, "WAPE": w}


def main():
    data = pd.read_parquet(OUT_DIR / "features.parquet")
    cutoff = data["date"].quantile(0.85)
    train = data[data["date"] < cutoff]
    test = data[data["date"] >= cutoff]
    print(f"[model] Train: {len(train):,} rows | Test: {len(test):,} rows "
          f"(cutoff {pd.Timestamp(cutoff).date()})")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    print("[model] Evaluation:")
    evaluate("Baseline(lag_7)", y_test, test["lag_7"].values)

    model = LGBMRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=8,
        num_leaves=50, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 0, None)
    evaluate("LightGBM", y_test, preds)

    joblib.dump(model, MODEL_DIR / "best_model_lgbm.pkl")
    joblib.dump(FEATURES, MODEL_DIR / "feature_list.pkl")
    imp = (pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
           .sort_values("importance", ascending=False))
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print("[model] Best model + feature importance saved.")


if __name__ == "__main__":
    main()
