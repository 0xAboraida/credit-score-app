import io
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

# ---------------------------------
# Streamlit Config
# ---------------------------------
st.set_page_config(page_title="Credit Score Classifier", layout="wide")
os.makedirs("uploads", exist_ok=True)

# ---------------------------------
# Load saved objects (same as API)
# ---------------------------------
# Keep names/paths exactly as your API
try:
    target_encoder = joblib.load('target_encoder.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    scaler = joblib.load('scaler.pkl')
    model = load_model('model_ANN2_full.keras')
    mlb = joblib.load('mlb.pkl')
    encoded_feature_names = joblib.load('encoded_feature_names.pkl')
    kmeans = joblib.load('kmeans_debt.pkl')
except Exception as e:
    st.error(f"⚠️ Failed to load artifacts: {e}")
    st.stop()

# ---------------------------------
# Mappings (identical to API)
# ---------------------------------
log_mapping = {
    "Annual_Income": "Log_Annual_Income",
    "Monthly_Inhand_Salary": "Log_Monthly_Inhand_Salary",
    "Outstanding_Debt": "Log_Outstanding_Debt",
    "Total_EMI_per_month": "Log_Total_EMI_per_month",
    "Amount_invested_monthly": "Log_Amount_invested_monthly",
    "Monthly_Balance": "Log_Monthly_Balance",
}

debt_mapping = {
    2: 0,  # Low Debt
    0: 1,  # Medium Debt
    3: 2,  # High Debt
    1: 3,  # Very High Debt
}

# ---------------------------------
# Preprocess (ported 1:1 from API)
# ---------------------------------

def preprocess_input(client_data):
    # Accept dict (single) or DataFrame (batch)
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
        df["Log_Income_Ratio"] = (np.log1p(df["Monthly_Inhand_Salary"] * 12)) - df["Log_Annual_Income"]

    # 3) MultiLabelBinarizer on Type_of_Loan_List
    if "Type_of_Loan_List" in df:
        df["Type_of_Loan_List"] = df["Type_of_Loan_List"].apply(lambda x: x if isinstance(x, list) else [])
        loans_transformed = mlb.transform(df["Type_of_Loan_List"])  # same classes/order as training
        loan_cols = mlb.classes_
        loans_transformed = pd.DataFrame(loans_transformed, columns=loan_cols, index=df.index)
        # API renames this exact column
        loans_transformed.rename(columns={"Not Specified": "Not Specified Loan"}, inplace=True)
        df = pd.concat([df, loans_transformed], axis=1)
    else:
        for loan in mlb.classes_:
            col_name = "Not Specified Loan" if loan == "Not Specified" else loan
            df[col_name] = 0

    # 4) Debt_Level via kmeans + mapping
    if "Outstanding_Debt" in df:
        cluster = kmeans.predict(df[["Outstanding_Debt"]])
        df["Debt_Level"] = pd.Series(cluster, index=df.index).map(debt_mapping)

    # 5) Drop unused original columns
    df.drop(
        [
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Outstanding_Debt",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance",
            "Type_of_Loan_List",
        ],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    # 6) Reorder columns using ordered_columns.json (same as API)
    with open("ordered_columns.json", "r") as f:
        ordered_columns = json.load(f)
    if "Credit_Score" in ordered_columns:
        ordered_columns.remove("Credit_Score")
    df = df[ordered_columns]

    # 7) Encode -> DataFrame with encoded_feature_names
    df_encoded_arr = preprocessor.transform(df)
    df_encoded = pd.DataFrame(df_encoded_arr, columns=encoded_feature_names, index=df.index)

    # 8) Scale
    df_scaled = scaler.transform(df_encoded)

    return df_scaled, df_encoded


# ---------------------------------
# Prediction helpers (mirror API outputs)
# ---------------------------------

def predict_single_to_api_like_payload(client_data: dict):
    X_scaled, X_encoded = preprocess_input(client_data)

    pred_probs = model.predict(X_scaled)
    pred_class = np.argmax(pred_probs, axis=1)
    pred_label = target_encoder.inverse_transform(pred_class)

    classes = list(target_encoder.classes_)
    probs_dict = {cls: float(pred_probs[0][i]) for i, cls in enumerate(classes)}

    encoded_dict = X_encoded.iloc[0].to_dict()
    scaled_dict = {col: float(val) for col, val in zip(X_encoded.columns, X_scaled[0])}

    # Return EXACTLY same structure the Flask API returned
    return {
        "prediction": pred_label[0],
        "probabilities": probs_dict,
        "encoded_before_scaling": encoded_dict,
        "scaled_for_model": scaled_dict,
    }


def predict_batch_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    # Same behavior as /predict_file endpoint
    X_scaled, _ = preprocess_input(df_in)

    pred_probs = model.predict(X_scaled)
    pred_class = np.argmax(pred_probs, axis=1)
    pred_label = target_encoder.inverse_transform(pred_class)

    df_out = df_in.copy()
    df_out["Prediction"] = pred_label

    # Add Prob columns with original encoder class order
    for j, cls in enumerate(target_encoder.classes_):
        df_out[f"Prob_{cls}"] = pred_probs[:, j]

    return df_out


# ---------------------------------
# THEME (copied from your GUI)
# ---------------------------------

def set_theme(theme):
    if theme == "Dark":
        css = """
        <style>
        .stApp { background: linear-gradient(135deg, #2c003e, #5e006a); color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .stButton {display: flex; justify-content: center;}
        .stButton>button { background: linear-gradient(135deg, #ff6ec7, #9b00ff); color: white; width:400px; height:65px; font-size:28px; padding: 0.6em 1.2em; font-weight: bold; border-radius: 12px; padding: 0.6em 1.2em; border: none; box-shadow: 2px 2px 8px rgba(0,0,0,0.4); transition: all 0.3s ease; }
        .stButton>button:hover { background: linear-gradient(135deg, #ff9ee1, #c34fff); color:black; transform: translateY(-2px); box-shadow: 3px 3px 12px rgba(0,0,0,0.6); }
        .stDataFrame, .stTable { background-color: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 10px; }
        h1, h2, h3, h4, h5, h6 { color: #ffd6f7; }
        label, .css-1vq4p4l { color: #ffffff !important; }
        div[data-testid="stNumberInput"], div[data-testid="stSelectbox"], div[data-testid="stTextInput"], div[data-testid="stMultiselect"] { border: 2px solid #a64ca6; border-radius: 12px; padding: 12px; margin-bottom: 15px; background: rgba(166,76,166,0.08); }
        label { font-weight: bold; color: white !important; margin-bottom: 5px; display: block; }
        ::placeholder { color: #d3d3d3 !important; }
        div[data-testid="stRadio"] > div > label > div { color: white !important; font-weight: bold; }
        div.stDownloadButton>button { color: purple !important; font-weight: bold; background-color:white; }
        </style>
        """
    else:
        css = """
        <style>
        .stApp { background: linear-gradient(135deg, #f0f2f6, #d9e2f0); color: #1a1a1a; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .stButton {display: flex; justify-content: center;}
        .stButton>button { background: linear-gradient(135deg, #ff6ec7, #9b00ff); color: white; width:400px; height:65px; font-size:28px; padding: 0.6em 1.2em; font-weight: bold; border-radius: 12px; padding: 0.6em 1.2em; border: none; box-shadow: 2px 2px 8px rgba(0,0,0,0.4); transition: all 0.3s ease; }
        .stButton>button:hover { background: linear-gradient(135deg, #4a90e2, #a3c4f5); color:black; transform: translateY(-2px); box-shadow: 3px 3px 12px rgba(0,0,0,0.5); }
        .stDataFrame, .stTable { background-color: rgba(0, 0, 0, 0.05); border-radius: 12px; padding: 10px; }
        h1, h2, h3, h4, h5, h6 { color: purple; }
        label, .css-1vq4p4l { color: #1a1a1a !important; }
        div[data-testid="stNumberInput"], div[data-testid="stSelectbox"], div[data-testid="stTextInput"], div[data-testid="stMultiselect"] { border: 2px solid purple; border-radius: 12px; padding: 12px; margin-bottom: 15px; background: linear-gradient(135deg, #d8b0f5, #a64ca6, #f0c3ff); }
        label { font-weight: bold; color: purple !important; margin-bottom: 5px; display: block; }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div>select, .stMultiselect>div>div>div>div { background-color: #ffffff !important; color: #1a1a1a !important; }
        ::placeholder { color: #a0a0a0 !important; }
        div.stDownloadButton>button { color: white !important; font-weight: bold; background-color:purple; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# ---------------------------------
# Header (copied)
# ---------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600;700&display=swap');
    h1 { font-family: 'Lora', sans-serif !important; font-weight: 700; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
    </style>
    <h1>Credit Score Classifier</h1>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------
# Top Controls (copied)
# ---------------------------------

theme = st.radio("Choose Theme", ["Dark", "Light"], horizontal=True, label_visibility="collapsed")
mode = st.radio("Choose Mode", ["Single Prediction", "Batch Prediction"], horizontal=True, label_visibility="collapsed")
set_theme(theme)

# Column titles (copied)
if mode == "Single Prediction":
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("<h3 style='text-align:center; color: purple;'>Enter Your Details</h3>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h3 style='text-align:center; color: purple;'>Your Credit Score Prediction</h3>", unsafe_allow_html=True)

# Display names (copied)
display_names = {
    "Month": "Month",
    "Age": "Age (Years)",
    "Occupation": "Occupation / Job",
    "Annual_Income": "Annual Income ($)",
    "Monthly_Inhand_Salary": "Monthly Salary ($)",
    "Num_Bank_Accounts": "Number of Bank Accounts",
    "Num_Credit_Card": "Number of Credit Cards",
    "Interest_Rate": "Interest Rate (%)",
    "Changed_Credit_Limit": "Credit Limit Changed ($)",
    "Num_Credit_Inquiries": "Credit Inquiries",
    "Credit_Mix": "Credit Mix",
    "Credit_Utilization_Ratio": "Credit Utilization (%)",
    "Delay_from_due_date": "Payment Delay (Days)",
    "Num_of_Delayed_Payment": "Delayed Payments Count",
    "Num_of_Loan": "Number of Loans",
    "Type_of_Loan_List": "Types of Loans",
    "Payment_of_Min_Amount": "Paid Minimum Amount?",
    "Payment_Behaviour": "Payment Behaviour",
    "Outstanding_Debt": "Outstanding Debt ($)",
    "Total_EMI_per_month": "Monthly EMI ($)",
    "Amount_invested_monthly": "Monthly Investments ($)",
    "Monthly_Balance": "Monthly Balance ($)",
    "Credit_History_by_Months": "Credit History (Months)",
}


# ---------------------------------
# Input fields (copied UI)
# ---------------------------------

def render_input_fields():
    col_input, col_output = st.columns([1.5, 1])
    inputs = {}
    with col_input:
        col1_input, col2_input, col3_input = st.columns(3)

        with col1_input:
            inputs["Month"] = st.selectbox(display_names["Month"], [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
            ])
            inputs["Age"] = st.number_input(display_names["Age"], 18, 100, 25)
            inputs["Occupation"] = st.selectbox(
                display_names["Occupation"],
                [
                    "Lawyer",
                    "Engineer",
                    "Architect",
                    "Mechanic",
                    "Scientist",
                    "Accountant",
                    "Developer",
                    "Media_Manager",
                    "Teacher",
                    "Entrepreneur",
                    "Doctor",
                    "Journalist",
                    "Manager",
                    "Musician",
                    "Writer",
                ],
            )
            inputs["Annual_Income"] = st.number_input(display_names["Annual_Income"], 0.0)
            inputs["Monthly_Inhand_Salary"] = st.number_input(display_names["Monthly_Inhand_Salary"], 0.0)
            inputs["Num_Bank_Accounts"] = st.number_input(display_names["Num_Bank_Accounts"], 0)
            inputs["Num_Credit_Card"] = st.number_input(display_names["Num_Credit_Card"], 0)

        with col2_input:
            inputs["Interest_Rate"] = st.number_input(display_names["Interest_Rate"], 0)
            inputs["Changed_Credit_Limit"] = st.number_input(display_names["Changed_Credit_Limit"], 0.0)
            inputs["Num_Credit_Inquiries"] = st.number_input(display_names["Num_Credit_Inquiries"], 0)
            inputs["Credit_Mix"] = st.selectbox(display_names["Credit_Mix"], ["Bad", "Standard", "Good"])
            inputs["Credit_Utilization_Ratio"] = st.number_input(display_names["Credit_Utilization_Ratio"], 0.0)
            inputs["Delay_from_due_date"] = st.number_input(display_names["Delay_from_due_date"], 0)
            inputs["Num_of_Delayed_Payment"] = st.number_input(display_names["Num_of_Delayed_Payment"], 0)

        with col3_input:
            inputs["Outstanding_Debt"] = st.number_input(display_names["Outstanding_Debt"], 0.0)
            inputs["Total_EMI_per_month"] = st.number_input(display_names["Total_EMI_per_month"], 0.0)
            inputs["Amount_invested_monthly"] = st.number_input(display_names["Amount_invested_monthly"], 0.0)
            inputs["Monthly_Balance"] = st.number_input(display_names["Monthly_Balance"], 0.0)
            inputs["Credit_History_by_Months"] = st.number_input(display_names["Credit_History_by_Months"], 0)
            inputs["Payment_of_Min_Amount"] = st.selectbox(display_names["Payment_of_Min_Amount"], ["Yes", "No"])
            inputs["Num_of_Loan"] = st.number_input(display_names["Num_of_Loan"], 0)

        col4_input, col5_input = st.columns(2)
        with col4_input:
            inputs["Payment_Behaviour"] = st.selectbox(
                display_names["Payment_Behaviour"],
                [
                    "High_spent_Large_value_payments",
                    "High_spent_Medium_value_payments",
                    "High_spent_Small_value_payments",
                    "Low_spent_Small_value_payments",
                    "Low_spent_Medium_value_payments",
                    "Low_spent_Large_value_payments",
                ],
            )
        with col5_input:
            inputs["Type_of_Loan_List"] = st.multiselect(
                display_names["Type_of_Loan_List"],
                [
                    "Auto Loan",
                    "Credit-Builder Loan",
                    "Debt Consolidation Loan",
                    "Home Equity Loan",
                    "Mortgage Loan",
                    "Not Specified",
                    "Payday Loan",
                    "Personal Loan",
                    "Student Loan",
                ],
            )

    return inputs, col_output


# ---------------------------------
# Results (copied UI)
# ---------------------------------

def render_results(result, col_output):
    with col_output:
        st.markdown(
            """
            <style>
            div.stButton > button:first-child { color: white; border: 1px solid #ccc; border-radius: 8px; font-weight: bold; }
            div.stButton > button:first-child:hover { background-color: white; color: black; border: 1px solid purple; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        if "prediction" in result:
            st.markdown(
                f"""
                <div style="text-align:center; margin-top:20px; margin-bottom:20px;">
                    <div style="background: linear-gradient(135deg, #8000ff, #ff00ff); border-radius:15px; color:white; padding:20px 30px; display:inline-block; font-size:24px; font-weight:700; box-shadow:0 8px 20px rgba(128,0,255,0.4);">
                        Prediction: <b>{result['prediction']}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if "probabilities" in result:
            probs_df = pd.DataFrame(list(result["probabilities"].items()), columns=["Credit_Score", "Probability"])
            fig_bar = px.bar(
                probs_df,
                x="Probability",
                y="Credit_Score",
                orientation="h",
                color="Probability",
                color_continuous_scale="Purples",
                text=(probs_df["Probability"] * 100).round(2),
            )
            fig_bar.update_layout(
                title={
                    "text": "<b>Prediction Probabilities (%)</b>",
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                    "font": {"size": 18, "color": "purple", "family": "Arial"},
                },
                xaxis_title="Probability (%)",
                yaxis_title="Credit Score",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            predicted_label = probs_df.loc[probs_df["Probability"].idxmax(), "Credit_Score"]
            predicted_value = probs_df["Probability"].max() * 100
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=predicted_value,
                    number={"font": {"size": 28, "color": "purple"}},
                    delta={"reference": 50, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                    gauge={"axis": {"range": [0, 100]}},
                )
            )
            fig_gauge.add_annotation(
                text=f"<b>Prediction:</b> {predicted_label}",
                x=0.5,
                y=1.3,
                showarrow=False,
                font={"size": 20, "color": "purple"},
                xanchor="center",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)


# ---------------------------------
# SINGLE MODE (no HTTP calls)
# ---------------------------------
if mode == "Single Prediction":
    client_data, col_output = render_input_fields()

    with col_output:
        if st.button("Predict Single", use_container_width=False):
            with st.spinner("⏳ Running prediction..."):
                try:
                    # Ensure loan list is list
                    client_data["Type_of_Loan_List"] = (
                        client_data.get("Type_of_Loan_List", []) if isinstance(client_data.get("Type_of_Loan_List", []), list) else []
                    )
                    result = predict_single_to_api_like_payload(client_data)
                except Exception as e:
                    result = {"prediction": f"API Error: {e}", "probabilities": {}}

            render_results(result, col_output)


# ---------------------------------
# BATCH MODE (no HTTP calls)
# ---------------------------------
else:
    st.markdown("<h3 style='text-align:center; color: purple;'>Upload Your CSV/Excel File</h3>", unsafe_allow_html=True)
    file = st.file_uploader("Upload your file", type=["csv", "xls", "xlsx"], label_visibility="collapsed")
    run_button = st.button("Run Batch Prediction", use_container_width=False)

    if file is not None and run_button:
        with st.spinner("⏳ Processing file..."):
            try:
                # Read exactly like API supports
                if file.name.endswith(".csv"):
                    df_in = pd.read_csv(file)
                elif file.name.endswith((".xls", ".xlsx")):
                    df_in = pd.read_excel(file)
                else:
                    st.error("Unsupported file type")
                    st.stop()

                df_preds = predict_batch_dataframe(df_in)

                st.markdown(
                    """
                    <div style="text-align:center; margin-top:20px; margin-bottom:20px;">
                        <div style="background: linear-gradient(135deg, #8000ff, #ff00ff); border-radius:15px; color:white; padding:20px 30px; display:inline-block; font-size:24px; font-weight:700; box-shadow:0 8px 20px rgba(128,0,255,0.4);">
                            <b>✅ Batch predictions ready! Here is a preview of the predictions:</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # a) Preview top 10 with same styling
                preview_df = df_preds.head(10)
                styler = preview_df.style
                # Try to color these columns if they exist
                prob_cols_expected = [f"Prob_{c}" for c in ["Good", "Standard", "Poor"]]
                prob_cols_present = [c for c in prob_cols_expected if c in preview_df.columns]
                if prob_cols_present:
                    styler = styler.background_gradient(subset=prob_cols_present, cmap="Purples")
                color_map = {"Good": "#1AC12E", "Poor": "#EC4B4B", "Standard": "#80D5CB"}
                if "Prediction" in preview_df.columns:
                    styler = styler.applymap(lambda x: f"background-color: {color_map.get(x, '')}", subset=["Prediction"]) 
                st.dataframe(styler)

                # b) Download predictions (same filename/mime as API)
                output = io.StringIO()
                df_preds.to_csv(output, index=False)
                st.download_button(
                    "Download Predictions",
                    data=output.getvalue(),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    key="download_button_upload",
                    icon="⬇️",
                )

                # c) Analysis (copied)
                st.markdown("<h3 style='text-align:center; color: purple;'> Analysis </h3>", unsafe_allow_html=True)

                col_plot1, col_plot2 = st.columns(2)
                with col_plot1:
                    fig_dist = px.histogram(
                        df_preds,
                        x="Prediction",
                        color="Prediction",
                        title="Distribution of Predicted Credit Scores",
                        color_discrete_sequence=["#610F60", "#A832A8", "#BF6BD9"],
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                    prob_cols = [c for c in df_preds.columns if c.startswith("Prob_")]
                    if prob_cols:
                        df_preds["Top_Prob"] = df_preds[prob_cols].max(axis=1)
                        fig_top_prob = px.box(
                            df_preds,
                            y="Top_Prob",
                            title="Top Probability (Confidence) Distribution",
                            color="Prediction",
                            color_discrete_sequence=["#610F60", "#A832A8", "#BF6BD9"],
                        )
                        st.plotly_chart(fig_top_prob, use_container_width=True)

                    fig_scatter = px.scatter(
                        df_preds,
                        x="Credit_Utilization_Ratio",
                        y="Outstanding_Debt",
                        color="Prediction",
                        size="Top_Prob" if "Top_Prob" in df_preds.columns else None,
                        hover_data=["Age", "Occupation", "Num_Credit_Card", "Num_Bank_Accounts"],
                        title="Credit Utilization vs Outstanding Debt",
                        color_discrete_sequence=["#610F60", "#A832A8", "#BF6BD9"],
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                    fig_scatter2 = px.scatter(
                        df_preds,
                        x="Outstanding_Debt",
                        y="Monthly_Inhand_Salary",
                        color="Prediction",
                        size="Interest_Rate",
                        hover_data=["Age", "Occupation", "Num_Credit_Card", "Num_Bank_Accounts"],
                        title="Debt vs Monthly Income per Credit Score (Point size indicates Interest Rate)",
                        color_discrete_sequence=["#9C0A99", "#BF40BF", "#D580D5"],
                    )
                    st.plotly_chart(fig_scatter2, use_container_width=True)

                with col_plot2:
                    score_counts = df_preds["Prediction"].value_counts().reset_index()
                    score_counts.columns = ["Credit_Score", "Count"]
                    fig_pie = px.pie(
                        score_counts,
                        names="Credit_Score",
                        values="Count",
                        title="Proportion of Predicted Credit Scores",
                        color="Credit_Score",
                        color_discrete_sequence=["#610F60", "#A832A8", "#BF6BD9"],
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    fig_violin = px.violin(
                        df_preds,
                        y="Top_Prob" if "Top_Prob" in df_preds.columns else prob_cols[0] if prob_cols else None,
                        x="Prediction",
                        color="Prediction",
                        box=True,
                        points="all",
                        title="Confidence Distribution per Credit Score",
                        color_discrete_sequence=["#610F60", "#A832A8", "#BF6BD9"],
                    )
                    st.plotly_chart(fig_violin, use_container_width=True)

                    fig_count = px.histogram(
                        df_preds,
                        x="Occupation",
                        color="Prediction",
                        barmode="group",
                        title="Predictions per Occupation",
                        color_discrete_sequence=["#610F60", "#A832A8", "#BF6BD9"],
                    )
                    st.plotly_chart(fig_count, use_container_width=True)

                    fig_scatter3 = px.scatter(
                        df_preds,
                        x="Outstanding_Debt",
                        y="Interest_Rate",
                        color="Prediction",
                        size="Age",
                        title="Outstanding Debt vs Interest Rate per Credit Score (Point size indicates Age)",
                        hover_data=["Age", "Occupation", "Num_Credit_Card", "Num_Bank_Accounts"],
                        color_discrete_sequence=["#9C0A99", "#BF40BF", "#D580D5"],
                    )
                    st.plotly_chart(fig_scatter3, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
