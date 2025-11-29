import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURES = ["active_users", "avg_latency_ms", "login_fail_rate", "pix_sucess_rate", "error_count"]

def train_model(csv_path="data/simulated_metric.csv", model_out="models/if_pipeline.joblib"):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    X = df[FEATURES].fillna(0)
    
    pipe = Pipeline([("scaler", StandardScaler()),
                      ("iforest", IsolationForest(n_estimators=200, contamination=0.01, random_state=42))])
    pipe.fit(X)
    
    #save
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, model_out)
    print(f"Saved model to {model_out}")
    return pipe

def load_model(model_out="models/if_pipeline.joblib"):
    return joblib.load(model_out)

if __name__ == "__main__":
    train_model()