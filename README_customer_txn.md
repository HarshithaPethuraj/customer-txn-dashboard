# 💳 Customer Transaction Prediction Dashboard

An end-to-end ML project predicting which bank customers will make a future transaction, built on the Santander dataset. Features a fully interactive React dashboard with model comparison, threshold tuning, and marketing ROI simulation.

🌐 **Live Demo:** [customer-txn-dashboard-app.vercel.app](https://customer-txn-dashboard-app.vercel.app)

---

## 🎯 Problem Statement

A bank wants to **identify which customers will make a specific transaction in the future**, irrespective of the amount transacted. The dataset is anonymized with 200 numerical features and a binary target:
- `target = 1` → customer WILL make the transaction
- `target = 0` → customer will NOT make the transaction

---

## ✨ What Makes This Project Unique

This is not a typical "just run XGBoost" approach. It combines:

- **Supervised modeling** — Logistic Regression, Random Forest, XGBoost, LightGBM
- **Unsupervised feature engineering** — PCA, KMeans clustering, and Isolation Forest anomaly scores added as engineered features to boost performance
- **Interactive React dashboard** — visualize everything from class imbalance to marketing ROI
- **Rich visual storytelling** — 15+ charts covering distributions, target separation, dimensionality reduction, clustering, and model diagnostics

---

## 📊 Dataset Overview

| Feature | Details |
|---------|---------|
| Rows | 200,000 customers |
| Columns | 202 (200 features + ID + target) |
| Target | Binary (0 = no transaction, 1 = will transact) |
| Class Balance | ~90% / 10% (heavily imbalanced) |
| Missing Values | None |
| Domain | Banking — Santander Customer Transaction |

---

## 🏆 Results

| Model | AUC Score |
|-------|-----------|
| LightGBM | **0.8856** ✅ Best |
| XGBoost | 0.8821 |
| Random Forest | 0.8634 |
| Logistic Regression | 0.8412 |

- 📈 **Top-Decile Lift: 5.56x** — targeting top 10% of scored customers captures 55.6% of all transactors
- 💰 **Marketing ROI** — model-targeted campaigns outperform mass outreach significantly

---

## 🖥️ Interactive Dashboard Features

- Analyze class imbalance and feature distributions across 200 variables
- Compare model performance side by side
- **Tune decision threshold** interactively — see real-time Precision / Recall / F1 impact
- Explore decile lift charts
- Simulate Marketing ROI — mass outreach vs model-targeted campaigns
- Review feature importance including engineered features

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| ML Models | LightGBM, XGBoost, Random Forest, Logistic Regression |
| Feature Engineering | PCA, KMeans, Isolation Forest |
| Libraries | scikit-learn, Pandas, NumPy, Matplotlib, Seaborn |
| Dashboard | React, Vite, Recharts |
| Deployment | Vercel |
| Notebook | Google Colab |

---

## 📁 Project Structure

```
├── Customer_Transaction_Prediction_final.ipynb   # Full ML notebook
├── src/                                           # React dashboard source
│   ├── components/                                # Dashboard components
│   └── App.jsx
├── package.json
└── README.md
```

---

## 🚀 Getting Started

**Run the notebook:**
- Open in Google Colab
- Mount Google Drive with dataset
- Run all cells

**Run the dashboard locally:**
```bash
npm install
npm run dev
```

---

## 👩‍💻 Author

**Harshitha Pethuraj** — AI & ML Engineer  
📧 harshithapethuraj@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/harshitha-pethuraj-13738129b/) | [GitHub](https://github.com/HarshithaPethuraj)
