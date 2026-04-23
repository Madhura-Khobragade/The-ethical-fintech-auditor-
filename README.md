# 🛡️ The Ethical Fintech Auditor: XAI for Loan Transparency

### 📌 Project Overview
In the financial world, AI models are often "black boxes." When a loan is rejected, users are left without a clear reason. This project solves that problem by using **Explainable AI (XAI)** to build a transparent loan eligibility system. 

The application doesn't just predict risk; it generates an **Adverse Action Report** using SHAP values, explaining exactly why a decision was made and providing an actionable plan for the user to improve their eligibility.

### 🚀 Key Features
- **Predictive Engine:** Built with **XGBoost** for high-accuracy credit risk assessment.
- **Transparency Layer:** Integrated **SHAP (SHapley Additive exPlanations)** to break down model weights into human-readable factors.
- **Actionable Feedback:** A "Counterfactual" logic system that suggests specific changes (e.g., reducing loan duration) to flip a rejection into an approval.
- **Web Interface:** A responsive **Flask** dashboard for real-time auditing.

### 🛠️ Tech Stack
- **Languages:** Python
- **Machine Learning:** XGBoost, Scikit-learn
- **Explainable AI:** SHAP
- **Web Framework:** Flask
- **Data:** Pandas, NumPy (German Credit Dataset)

### ⚖️ Ethical Compliance
This project aligns with the **GDPR "Right to Explanation"** and the **Fair Credit Reporting Act (FCRA)**, demonstrating how AI can be both powerful and accountable.

### 🔧 How to Run
1. Clone the repository.
2. Install dependencies: `pip install flask xgboost shap pandas scikit-learn`
3. Run the application: `python app.py`
4. Access the dashboard at `http://127.0.0.1:5000`