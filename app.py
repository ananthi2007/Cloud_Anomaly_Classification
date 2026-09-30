import streamlit as st
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Cloud Resource Anomaly Detection",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ Cloud Service Resource Anomaly Detection")
st.write(
    "Enter cloud-service resource metrics to predict whether "
    "the observation is normal or anomalous."
)

# Load dataset
df = pd.read_csv("test.csv")

# Features and target
X = df.drop(columns=["timestamp", "label", "event_id"])
y = df["label"]

# Train model
model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42,
    max_depth=6
)

model.fit(X, y)

st.subheader("Enter Resource Metrics")

input_data = {}

cols = st.columns(3)

for i, feature in enumerate(X.columns):
    with cols[i % 3]:
        input_data[feature] = st.number_input(
            feature,
            value=float(X[feature].mean())
        )

if st.button("🔍 Predict Anomaly"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.error("⚠️ ANOMALY DETECTED")
    else:
        st.success("✅ NORMAL")

    st.metric(
        "Anomaly Probability",
        f"{probability * 100:.2f}%"
    )
