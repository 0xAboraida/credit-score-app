import numpy as np
import pandas as pd
import json
from flask import Flask, request, jsonify, make_response
import joblib
from tensorflow.keras.models import load_model
import io
import os


# --------------------------
# Load saved objects
# --------------------------
target_encoder = joblib.load('target_encoder.pkl')
preprocessor = joblib.load("preprocessor.pkl")
scaler = joblib.load("scaler.pkl")
model = load_model("model_ANN2_full.keras")
mlb = joblib.load("mlb.pkl")  
encoded_feature_names = joblib.load("encoded_feature_names.pkl")  
kmeans = joblib.load("kmeans_debt.pkl")  

# --------------------------
# Log transform mapping
# --------------------------
log_mapping = {
    "Annual_Income": "Log_Annual_Income",
    "Monthly_Inhand_Salary": "Log_Monthly_Inhand_Salary",
    "Outstanding_Debt": "Log_Outstanding_Debt",
    "Total_EMI_per_month": "Log_Total_EMI_per_month",
    "Amount_invested_monthly": "Log_Amount_invested_monthly",
    "Monthly_Balance": "Log_Monthly_Balance"
}

# --------------------------
# Debt_Level mapping
# --------------------------
debt_mapping = {
    2: 0,  # Low Debt
    0: 1,  # Medium Debt
    3: 2,  # High Debt
    1: 3   # Very High Debt
}

# --------------------------
# Flask App
# --------------------------
app = Flask(__name__)

# --------------------------
# Preprocessing function
# --------------------------
def preprocess_input(client_data):
    if isinstance(client_data, dict):
        df = pd.DataFrame([client_data])
    else:
        df = client_data.copy()

    # 1) log transforms
    for original, log_col in log_mapping.items():
        if original in df:
            df[log_col] = np.log1p(df[original])

    # 2) Log_Income_Ratio
    if "Monthly_Inhand_Salary" in df and "Log_Annual_Income" in df:
        df['Log_Income_Ratio'] = (np.log1p(df['Monthly_Inhand_Salary'] * 12)) - df['Log_Annual_Income']

    # 3) MultiLabelBinarizer
    if "Type_of_Loan_List" in df:
        df["Type_of_Loan_List"] = df["Type_of_Loan_List"].apply(lambda x: x if isinstance(x, list) else [])
        loans_transformed = mlb.transform(df["Type_of_Loan_List"])
        loan_cols = mlb.classes_
        loans_transformed = pd.DataFrame(loans_transformed, columns=loan_cols, index=df.index)
        loans_transformed.rename(columns={"Not Specified": "Not Specified Loan"}, inplace=True)
        df = pd.concat([df, loans_transformed], axis=1)
    else:
        for loan in mlb.classes_:
            col_name = "Not Specified Loan" if loan == "Not Specified" else loan
            df[col_name] = 0

    # 4) Debt_Level
    if "Outstanding_Debt" in df:
        cluster = kmeans.predict(df[["Outstanding_Debt"]])
        df["Debt_Level"] = pd.Series(cluster).map(debt_mapping)

    # 5) Drop unused
    df.drop(["Annual_Income", "Monthly_Inhand_Salary", "Outstanding_Debt",
             "Total_EMI_per_month", "Amount_invested_monthly", 
             "Monthly_Balance", "Type_of_Loan_List"], axis=1, inplace=True, errors='ignore')

    # 6) Reorder
    with open("ordered_columns.json", "r") as f:
        ordered_columns = json.load(f)
    if "Credit_Score" in ordered_columns:
        ordered_columns.remove("Credit_Score")
    df = df[ordered_columns]

    # 7) Encode
    df_encoded = preprocessor.transform(df)
    df_encoded = pd.DataFrame(df_encoded, columns=encoded_feature_names, index=df.index)

    # 8) Scale
    df_scaled = scaler.transform(df_encoded)

    return df_scaled, df_encoded



# --------------------------
# Run (for local server)
# --------------------------
@app.route('/', methods=['GET'])
def home():
    return "Hello Maaaan!"

# --------------------------
# Prediction Endpoint
# --------------------------
@app.route('/predict', methods=['POST'])
def predict():
    client_data = request.get_json()
    X_scaled, X_encoded = preprocess_input(client_data)

    # --- Prediction ---
    pred_probs = model.predict(X_scaled)
    pred_class = np.argmax(pred_probs, axis=1)
    pred_label = target_encoder.inverse_transform(pred_class)
    classes = target_encoder.classes_
    probs_dict = {cls: float(pred_probs[0][i]) for i, cls in enumerate(classes)}

    # --- Encoded / Scaled dicts ---
    encoded_dict = X_encoded.iloc[0].to_dict()
    scaled_dict = {col: float(val) for col, val in zip(X_encoded.columns, X_scaled[0])}

    return jsonify({
        "prediction": pred_label[0],
        "probabilities": probs_dict,
        "encoded_before_scaling": encoded_dict,
        "scaled_for_model": scaled_dict
    })


from werkzeug.utils import secure_filename
import os

# --------------------------
# Prediction from File
# --------------------------
@app.route('/predict_file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # --- Read File ---
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    # --- Preprocess once ---
    X_scaled, _ = preprocess_input(df)

    # --- Predictions ---
    pred_probs = model.predict(X_scaled)
    pred_class = np.argmax(pred_probs, axis=1)
    pred_label = target_encoder.inverse_transform(pred_class)

    # --- Add results to DataFrame ---
    df["Prediction"] = pred_label

    # We add the probabilities of each class as columns.
    for j, cls in enumerate(target_encoder.classes_):
        df[f"Prob_{cls}"] = pred_probs[:, j]


    # --- CSV output  
    output = io.StringIO()
    df.to_csv(output, index=False)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    response.headers["Content-Type"] = "text/csv"
    return response




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), debug=False)
