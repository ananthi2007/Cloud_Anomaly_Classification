# Cloud Service Resource Anomaly Classification

## 1. Project Overview

Cloud services continuously generate large amounts of resource-usage data. Unusual patterns in CPU usage, memory usage, disk I/O, network traffic, API latency, and other cloud metrics may indicate abnormal system behaviour.

This project develops a machine learning based system to classify cloud-service resource observations as either:

- Normal
- Anomalous

The project uses the AIClusterKPI dataset and compares multiple machine learning classification models.

---

## 2. Problem Statement

The objective is to identify unusual resource-usage observations in a simulated cloud-service environment using machine learning.

The system takes cloud resource KPI measurements as input and predicts whether the observation is normal or anomalous.

---

## 3. Dataset

Dataset: AIClusterKPI: A Multivariate KPI Dataset for Cloud-Native Anomaly Detection

Source: Zenodo

Dataset DOI: 10.5281/zenodo.19926114

The dataset contains cloud-native KPI measurements including:

- CPU utilization
- Available memory
- Disk I/O wait
- Network inbound traffic
- Pending pods
- API server latency
- Container restarts

The `test.csv` file contains 3,029 observations and 31 KPI features used for the classification experiment.

Class distribution:

- Normal: 1,912
- Anomaly: 1,117

---

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- VS Code

---

## 5. Machine Learning Models

Three classification models were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

## 6. Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6733 | 0.5566 | 0.5516 | 0.5541 | 0.7114 |
| Decision Tree | 0.5809 | 0.4563 | 0.7265 | 0.5606 | 0.6757 |
| KNN | 0.7096 | 0.6741 | 0.4081 | 0.5084 | 0.7148 |

The models show different performance characteristics. The Decision Tree achieved higher anomaly recall, while KNN achieved higher accuracy and ROC-AUC in this experiment.

---

## 7. Streamlit Application

A Streamlit web application was developed to provide an interactive anomaly prediction interface.

The application accepts cloud KPI values as input and predicts:

- Normal
- Anomaly

It also displays the predicted anomaly probability.

Run the application using:

```bash
python -m streamlit run app.py