import time
import pandas as pd
from datetime import datetime
from src.model import load_model, FEATURES

def stream_and_alert(csv_path="data/simulated_metrics.csv", model_path="models/if_pipeline.joblib", interval=0.01):
    pipe = load_model(model_path)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    alerts = []

    for _, row in df.iterrows():
        X = row[FEATURES].values.reshape(1, -1)
        is_inlier = pipe.predict(X)[0] # 1 inlier, -1 outlier
        score = pipe.named_steps["iforest"].decision_function(X)[0]

        if is_inlier == -1:
            alert = {
                "timestamp": str(row["timestamp"]),
                "active_users": int(row["active_users"]),
                "avg_latency_ms": float(row["avg_latency_ms"]),
                "login_fail_rate": float(row["login_fail_rate"]),
                "pix_sucess_rate": float(row["pix_sucess_rate"]),
                "error_count": int(row["error_count"]),
                "score": float(score),
                "severity": "high" if row["avg_latency_ms"] > 1000 or row["error_count"] > 20 else "medium"
                }
            
            print("ALERT:", alert)
            alerts.append(alert)
        time.sleep(interval)
    return alerts

if __name__ == "__main__":
    stream_and_alert()