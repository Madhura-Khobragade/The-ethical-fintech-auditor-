from flask import Flask, render_template
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# --- PREVIOUS LOGIC (Data & Training) ---
df = pd.read_csv('archive/german_credit_data.csv', index_col=0).fillna('unknown')
target_col = 'target'
encoders = {}

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

X = df.drop(target_col, axis=1)
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(eval_metric='logloss', n_estimators=50).fit(X_train, y_train)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# --- WEB ROUTE ---
@app.route('/')
def home():
    # Let's audit Customer 0 for the demo
    index = 0
    person_data = X_test.iloc[[index]]
    pred_code = model.predict(person_data)[0]
    prob = model.predict_proba(person_data)[0]
    
    # Determine label
    label = encoders[target_col].inverse_transform([pred_code])[0] if target_col in encoders else ('bad' if pred_code == 0 else 'good')
    
    # XAI logic
    current_shap = shap_values[index] if len(shap_values.shape) == 2 else shap_values[index, :, 1]
    feature_impacts = pd.Series(current_shap, index=X_test.columns)
    top_reason = feature_impacts.sort_values(ascending=True).index[0]
    
    # Build result object for the UI
    result = {
        "decision": "REJECTED" if str(label).lower() == 'bad' else "APPROVED",
        "factor": top_reason,
        "explanation": f"The model identified '{top_reason}' as the most significant risk factor.",
        "confidence": f"{prob[pred_code]:.2f}",
        "advice": f"Reducing your {top_reason.replace('_', ' ')} may increase your approval chances."
    }
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)