import streamlit as st
import pandas as pd
from src.model import load_model, FEATURES

st.set_page_config(layout="wide", page_title="AIOps Customer Experience Monitor")

@st.cache_data
def load_data(path="data/simulated_metrics.csv"):
    return pd.read_csv(path, parse_dates=["timestamp"])

def detect_anomalies(df, pipe):
    X = df[FEATURES].fillna(0)
    preds = pipe.predict(X)
    scores = pipe.named_steps["iforest"].decision_function(X)

    df2 =df.copy()
    df2["is_anomaly"] = (preds == -1)
    df2["score"] = scores
    return df2

def main():
    st.title("AIOps - Customer Exprerience Anomaly Monitor")
    df = load_data()
    st.sidebar.header("Config")
    model = load_model()
    df2 = detect_anomalies(df, model)

    st.metric("Total rows", len(df2))
    st.metric("Anomalias detectadas", int(df2["is_anomaly"].sum()))

    st.subheader("Séries temporais")

    st.line_chart(df2.set_index("timestamp")[["avg_latency_ms", "error_count"]])

    st.subheader("Anomalias")

    st.dataframe(df2[df2["is_anomaly"]].sort_values("timestamp", ascending=False).head(200))

    if __name__ == "__main__":
        main()