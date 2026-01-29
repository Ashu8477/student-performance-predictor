from flask import Flask, render_template, request, jsonify
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)

# =========================
# Encoding Function
# =========================
def encode_data(df):
    mappings = {
        'higher': {'no': 0, 'yes': 1},
        'internet': {'no': 0, 'yes': 1},
        'passed': {'no': 0, 'yes': 1}
    }
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

# =========================
# Load & Prepare Data
# =========================
df = pd.read_csv("student-data.csv")
df = encode_data(df)

# IMPORTANT: only those features which UI uses
FEATURES = [
    "studytime",
    "failures",
    "goout",
    "higher",
    "internet",
    "Medu",
    "Fedu"
]

X = df[FEATURES]
y = df["passed"]

# =========================
# Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Train Model (Random Forest)
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# =========================
# Metrics
# =========================
y_pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
f1 = round(f1_score(y_test, y_pred), 2)

# =========================
# Routes
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/metrics")
def metrics():
    return jsonify({
        "accuracy": accuracy,
        "f1": f1
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Build input EXACTLY like training
    input_df = pd.DataFrame([{
        "studytime": int(data["studytime"]),
        "failures": int(data["failures"]),
        "goout": int(data["goout"]),
        "higher": int(data["higher"]),
        "internet": int(data["internet"]),
        "Medu": int(data["Medu"]),
        "Fedu": int(data["Fedu"]),
    }])

    prob = model.predict_proba(input_df)[0][1]

    return jsonify({
        "result": "PASS" if prob >= 0.5 else "FAIL",
        "confidence": round(prob * 100, 2)
    })

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)
