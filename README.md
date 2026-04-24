# 🏏 IPL Match Winner Predictor — Machine Learning + Streamlit App

A production-grade **Machine Learning web application built using Streamlit** that predicts the probability of a team winning an IPL match in real-time based on match conditions and historical data.

This project demonstrates a complete **end-to-end ML pipeline with interactive deployment**, combining data science and user-centric UI.

---

## 🌐 Live Preview

👉 Try the app here:  
🔗 https://ipl-match-outcome-predictor-bymeet.streamlit.app/

---

## 🚀 Overview

The IPL Predictor uses trained ML models on historical IPL match data to dynamically compute **win probabilities** during a live match scenario.

It captures real-world match dynamics such as:

* Required run rate pressure
* Wicket loss impact
* Target chasing complexity
* Team performance trends

---

## 🧠 Key Features

### 🔹 Real-Time Win Probability Prediction

* Predicts match outcome using:

  * Batting Team
  * Bowling Team
  * Current Score
  * Overs Completed
  * Wickets Fallen
  * Target Score

---

### 🔹 Interactive Streamlit UI

* Clean and responsive interface
* Instant predictions without page reload
* Slider & dropdown-based inputs
* Real-time probability visualization

---

### 🔹 Feature Engineering

* Current Run Rate (CRR)
* Required Run Rate (RRR)
* Balls Remaining
* Match Pressure Index
* Team Encoding

---

### 🔹 Machine Learning Models

* Logistic Regression
* Random Forest
* Gradient Boosting *(final model)*

Model selected based on:

* Accuracy
* Generalization
* Stability across datasets

---

## 🛠️ Tech Stack

### 📊 Machine Learning

* Python
* Scikit-learn
* Pandas
* NumPy

---

### 🌐 Frontend (Streamlit)

* Streamlit
* Custom UI Components

---

### ⚙️ Tools

* Jupyter Notebook
* Git & GitHub
* VS Code

---

## 📂 Project Workflow

```id="u1txfk"
Data Collection → Data Cleaning → Feature Engineering → Model Training → Evaluation → Deployment (Streamlit)
```

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash id="sl2v1y"
git clone <YOUR_GIT_URL>
cd IPL-Predictor
```

---

### 2️⃣ Install Dependencies

```bash id="n3mdfx"
pip install -r requirements.txt
```

---

### 3️⃣ Run the Streamlit App

```bash id="p2v9qn"
streamlit run app.py
```

---

## 📊 Input Parameters

| Feature         | Description            |
| --------------- | ---------------------- |
| Batting Team    | Team currently batting |
| Bowling Team    | Opponent team          |
| Current Score   | Runs scored            |
| Overs Completed | Overs played           |
| Wickets Fallen  | Wickets lost           |
| Target Score    | Total target           |

---

## 📈 Output

* 🎯 Win Probability (%)
* 📊 Confidence estimation

---
## Demo Link
https://ipl-match-outcome-predictor-bymeet.streamlit.app/

## 🧪 Model Details

* Algorithm: **Gradient Boosting Classifier**
* Dataset: Historical IPL matches
* Evaluation Metrics:

  * Accuracy
  * Confusion Matrix
  * ROC-AUC Score

---

## 📁 Project Structure

```id="h6sd8f"
IPL-Predictor/
│
├── data/                # Dataset
├── notebooks/           # EDA & training
├── model/               # Saved ML model (.pkl)
├── app.py               # Streamlit application
├── utils/               # Feature engineering functions
├── requirements.txt
└── README.md
```

---

## 🔮 Future Enhancements

* Live match API integration
* Deep Learning models (LSTM)
* Player-level analytics
* Deployment on Streamlit Cloud / AWS
* Real-time data ingestion

---

## ⚠️ Disclaimer

This application is intended for **educational and analytical purposes only**. Predictions are based on historical data and may not reflect actual match outcomes.

---

## 📄 License

Licensed under the **MIT License**.
