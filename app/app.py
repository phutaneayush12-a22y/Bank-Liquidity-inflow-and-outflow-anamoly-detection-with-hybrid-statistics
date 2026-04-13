import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Liquidity Dashboard", layout="wide")

# -------------------------------
# HEADER
# -------------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
    🏦 Bank Liquidity Anomaly Detection Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# LOAD MODEL & SCALER
# -------------------------------
model = joblib.load("../models/isolation_forest.pkl")
scaler = joblib.load("../models/scaler.pkl")

features = list(scaler.feature_names_in_)

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("../DATA/Processed_data/df_final.csv")

df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# -------------------------------
# INIT HISTORY
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = df['net_flow'].tail(50).tolist()

history = st.session_state.history

# -------------------------------
# CONTROLS
# -------------------------------
colA, colB = st.columns([2, 1])

with colA:
    new_value = st.number_input("Enter Net Flow", value=13000.0)

with colB:
    if st.button("🔄 Reset History"):
        st.session_state.history = df['net_flow'].tail(50).tolist()
        st.success("History reset")

# -------------------------------
# PREDICT
# -------------------------------
if st.button("Predict"):

    # Update history
    history.append(new_value)
    if len(history) > 50:
        history.pop(0)

    # -------------------------------
    # DATA PREP
    # -------------------------------
    hist_df = pd.DataFrame(history, columns=['net_flow'])
    hist_df['net_flow'] = hist_df['net_flow'].clip(0, 80000)

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    hist_df['lag_1'] = hist_df['net_flow'].shift(1)
    hist_df['lag_24'] = hist_df['net_flow'].shift(24)

    hist_df['rolling_mean_24'] = hist_df['net_flow'].rolling(24).mean()
    hist_df['rolling_std_24'] = hist_df['net_flow'].rolling(24).std()

    hist_df['net_flow_diff'] = hist_df['net_flow'].diff()
    hist_df['pct_change'] = hist_df['net_flow'].pct_change()

    hist_df['hour'] = np.arange(len(hist_df)) % 24
    hist_df['hour_sin'] = np.sin(2*np.pi*hist_df['hour']/24)
    hist_df['hour_cos'] = np.cos(2*np.pi*hist_df['hour']/24)
    hist_df['is_weekend'] = 0

    hist_df['interest_rate'] = df['interest_rate'].iloc[-1]

    hist_df = hist_df.dropna()

    # -------------------------------
    # MODEL PREDICTION
    # -------------------------------
    if len(hist_df) == 0:
        st.warning("Not enough data")
    else:
        latest = hist_df.iloc[-1:][features]
        latest_scaled = scaler.transform(latest)

        score = model.decision_function(latest_scaled)[0]

        # -------------------------------
        # IQR NORMAL ZONE
        # -------------------------------
        q1 = np.percentile(history, 25)
        q3 = np.percentile(history, 75)
        iqr = q3 - q1

        lower = max(q1 - 1.5 * iqr, 0)
        upper = min(q3 + 1.5 * iqr, 80000)

        # -------------------------------
        # FINAL DECISION
        # -------------------------------
        if new_value < 2000 or new_value > 80000:
            result = "🚨 Anomaly (Extreme Value)"
        elif new_value < lower or new_value > upper:
            result = "🚨 Anomaly (Out of Normal Range)"
        elif score < -0.15:
            result = "🚨 Anomaly (Pattern Deviation)"
        else:
            result = "✅ Normal"

        # -------------------------------
        # KPI CARDS
        # -------------------------------
        st.markdown("## 📊 Detection Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Status")
            if "Anomaly" in result:
                st.error("Anomaly")
            else:
                st.success("Normal")

        with col2:
            st.markdown("### Score")
            st.info(f"{score:.4f}")

        with col3:
            st.markdown("### Typical Zone")
            st.info(f"{int(q1)} – {int(q3)}")

        # -------------------------------
        # EXPLANATION
        # -------------------------------
        st.markdown("### 🧠 Explanation")

        if "Extreme" in result:
            st.warning("Value is extremely high or low.")
        elif "Range" in result:
            st.warning("Value is outside the normal operating range.")
        elif "Pattern" in result:
            st.warning("Unusual pattern detected by model.")
        else:
            st.success("Value follows normal behavior.")

        # -------------------------------
        # EXTRA INFO
        # -------------------------------
        st.write(f"Allowed Range: {int(lower)} to {int(upper)}")
        st.write("Recent Values:", history[-5:])

# -------------------------------
# GRAPH
# -------------------------------
st.markdown("## 📈 Liquidity Trend")

chart_df = pd.DataFrame(history, columns=["Net Flow"])

st.line_chart(chart_df)

# Normal zone display
if len(history) > 10:
    q1 = np.percentile(history, 25)
    q3 = np.percentile(history, 75)

    st.markdown(f"🟢 Typical Normal Zone: {int(q1)} to {int(q3)}")
    st.markdown("🔴 Values outside this range may indicate anomalies")