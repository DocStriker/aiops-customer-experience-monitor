import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_base_series(start_time, minutes):
    rng = [start_time + timedelta(minutes=i) for i in range(minutes)]

    active_users = np.random.poisson(lam=4000, size=minutes)

    avg_latency_ms = np.random.normal(loc=180, scale=30, size=minutes).clip(min=50)

    login_fail_rate = np.random.beta(a=1.2, b=2, size=minutes) * 100

    pix_sucess_rate = np.random.beta(a=800, b=2, size=minutes) * 100

    error_count = np.random.poisson(lam=3, size=minutes)

    return pd.DataFrame({
        "timestamp": rng,
        "active_users": active_users,
        "avg_latency_ms":avg_latency_ms,
        "login_fail_rate": login_fail_rate,
        "pix_sucess_rate": pix_sucess_rate,
        "error_count": error_count
    })

def inject_anomalies(df, seed=42):
    np.random.seed(seed)
    n = len(df)

    anomaly_points = np.random.choice(np.arange(60, n-60), size=max(3, n//300), replace=False)

    for pt in anomaly_points:
        typ = np.random.choice(["latency_spike", "login_fail", "pix_drop", "error_burst"])

        if typ == "latency_spike":
            duration = np.random.randint(3, 20)
            df.loc[pt:pt+duration, "avg_latency_ms"] *= np.random.uniform(5, 15)
            df.loc[pt:pt+duration, "error_count"] += np.random.poisson(lam=20, size=duration+1)

        elif typ == "login_fail":
            duration = np.random.randint(2, 15)
            df.loc[pt:pt+duration, "login_fail_rate"] *= np.random.uniform(10, 60)
            df.loc[pt:pt+duration, "error_count"] += np.random.poisson(lam=10, size=duration+1)

        elif typ == "pix_drop":
            duration = np.random.randint(2, 15)
            df.loc[pt:pt+duration, "pix_sucess_rate"] *= np.random.uniform(0.1, 0.6)
            df.loc[pt:pt+duration, "error_count"] += np.random.poisson(lam=15, size=duration+1)

        elif typ == "error_burst":
            duration = np.random.randint(1, 6)
            df.loc[pt:pt+duration, "avg_latency_ms"] *= np.random.uniform(2, 6)
            df.loc[pt:pt+duration, "error_count"] += np.random.poisson(lam=50, size=duration+1)

        df["pix_sucess_rate"] = df["pix_sucess_rate"].clip(0, 100)
        df["login_fail_rate"] = df["login_fail_rate"].clip(0, 100)
        df["avg_latency_ms"] = df["avg_latency_ms"].clip(lower=10)

        return df
    
def generate_and_save(minutes=24*60, out_path="../data/simulated_metrics.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    start_time = datetime.utcnow() - timedelta(minutes=minutes)

    df = generate_base_series(start_time, minutes)
    df = inject_anomalies(df)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    generate_and_save(minutes=24*60*3)