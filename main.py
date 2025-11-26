import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import datetime
from collections import deque
import xgboost as xgb
import os 
from google import genai
from google.genai import types
import matplotlib.pyplot as plt 
import time # <-- ADDED FOR EXPONENTIAL BACKOFF

# --- CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="CollabBoard PCOS Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants (Must match your training setup)
WINDOW = 30 # ROLLING_WINDOW
SEED = 42
DAILY_FEATURES = [
    'WeightKg', 'BMI',
    'PimplesYN', 'hairgrowthYN', 'SkindarkeningYN',
    'HairlossYN', 'FastfoodYN', 'RegExerciseYN'
]
STATIC_FEATURES_TO_USE = ['Ageyrs', 'HeightCm']

# --- CACHING FUNCTIONS (Optimization) ---

@st.cache_resource
def load_assets():
    """Loads the ML assets and initializes the Gemini client."""
    
    # 1. Initialize Gemini Client
    client = None
    try:
        # Streamlit automatically loads secrets.toml into st.secrets.
        # This is the correct way to load secrets in a Streamlit app.
        api_key = st.secrets.get("GEMINI_API_KEY")
        
        if api_key and api_key != "YOUR_GEMINI_API_KEY_HERE":
            client = genai.Client(api_key=api_key)
        else:
            st.warning("⚠️ Gemini API Key not found or set to placeholder in secrets.toml. AI explanation feature will be disabled.")
            
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        
    # 2. Load ML assets
    try:
        # Load the trained model using the native XGBoost method (assuming .json format fix)
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model('streamlit_assets/xgb_model.json')
        
        # Load the list of features used in training
        X_train_cols = joblib.load('streamlit_assets/feature_columns.joblib')
        
        # Initialize the SHAP Explainer from the loaded model
        explainer = shap.TreeExplainer(xgb_model)
        
        # Load the base data
        base_df = pd.read_csv('streamlit_assets/base_patient_data.csv')

        return xgb_model, explainer, X_train_cols, base_df, client
    
    except FileNotFoundError as e:
        st.error(f"Error loading required files: {e}. "
                 "Ensure all assets are in the 'streamlit_assets' directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during ML asset loading: {e}")
        st.stop()

@st.cache_data
def get_initial_history(base_df, patient_id=1):
    """Gets the history for a single patient from the base data."""
    # This logic seeds the history deque for the rolling window calculation.
    patient_data = base_df[base_df['PatientID'] == patient_id].iloc[0].to_dict()
    history = deque(maxlen=WINDOW)

    for day in range(WINDOW):
        entry = {
            'Date': datetime.date.today() - datetime.timedelta(days=WINDOW - 1 - day),
            'PatientID': patient_id
        }
        for feature in DAILY_FEATURES:
            entry[feature] = patient_data[feature]
        
        history.append(entry)
    
    return history, patient_data


# --- AI EXPLANATION FUNCTION ---

def generate_shap_explanation(X_new, shap_values, expected_value, prediction_score):
    """
    Sends SHAP data to the Gemini API to get a natural language explanation,
    with exponential backoff for transient errors (e.g., 503 UNAVAILABLE).
    """
    # Use the globally available client object
    if not client:
        return "⚠️ Error: AI explanation disabled (API Key missing)."

    # 1. Prepare data into a simple, readable format for the prompt
    feature_data = []
    # shap_values[0] accesses the array of values for the current prediction
    sorted_indices = np.argsort(np.abs(shap_values[0]))[::-1]
    
    for i in sorted_indices:
        feature_name = X_new.columns[i]
        feature_value = X_new.iloc[0, i]
        contribution = shap_values[0][i]
        
        feature_data.append(
            f"- **{feature_name}** (Value: {feature_value:.2f}): {contribution:+.3f} ({"Increased Risk" if contribution > 0 else "Decreased Risk"})"
        )
    
    data_summary = "\n".join(feature_data[:6]) # Focus on top 6 contributors

    # 2. Construct the detailed prompt
    system_instruction = (
        "You are an empathetic, professional medical assistant. Your task is to interpret "
        "a machine learning prediction for a patient's PCOS risk based on SHAP values. "
        "Explain the result in simple, non-technical language. Focus on the key factors "
        "that most significantly increased and decreased the predicted risk. Start by "
        "explaining the overall prediction score."
    )

    user_prompt = f"""
    The model predicted a **PCOS Risk Score of {prediction_score:.2f}** (0.00 to 1.00).
    The average risk (Base Value) is {expected_value:.2f}.
    
    Here are the top contributing features and their SHAP contributions:
    {data_summary}

    Please provide a concise, two-paragraph explanation:
    1. Summarize the overall result and what it means (e.g., higher or lower than average risk).
    2. Detail the 2-3 most important factors that pushed the score higher (risk factors) and the 2-3 most important factors that pushed the score lower (protective factors).
    """

    # 3. Call the Gemini API with Exponential Backoff
    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            )
            return response.text
        
        except Exception as e:
            # Check specifically for UNAVAILABLE or other transient errors (500s)
            error_message = str(e)
            if attempt < MAX_RETRIES - 1 and ("503" in error_message or "UNAVAILABLE" in error_message):
                wait_time = 2 ** attempt  # Exponential wait: 1s, 2s, 4s, 8s...
                st.warning(f"Gemini API temporarily unavailable (503). Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                return f"Gemini API call failed permanently after {attempt + 1} attempts: {e}"

# --- SESSION STATE INITIALIZATION ---
# Load all assets
xgb_model, explainer, X_train_cols, base_df, client = load_assets()

if 'history' not in st.session_state:
    # Initialize history deque and base patient features
    st.session_state['history'], st.session_state['base_features'] = get_initial_history(base_df)

if 'last_entry_date' not in st.session_state:
    st.session_state['last_entry_date'] = st.session_state['history'][-1]['Date']

if 'streak' not in st.session_state:
    st.session_state['streak'] = 0


# --- FEATURE ENGINEERING & PREDICTION LOGIC ---

def prepare_and_predict(new_entry_dict, base_features, history_deque, X_cols):
    """Calculates rolling features, prepares the final vector, and predicts."""

    # 1. Combine History for Rolling Calculation
    # Convert deque to DataFrame for easy rolling calculation
    temp_df = pd.DataFrame(history_deque)
    temp_df = pd.concat([temp_df, pd.DataFrame([new_entry_dict])], ignore_index=True)
    
    # 2. Calculate Rolling Features 
    
    # Continuous Rolling Average
    ROLLING_COLS = ['WeightKg', 'BMI']
    for col in ROLLING_COLS:
        col_name = f'{col}_Avg_{WINDOW}d'
        rolling_avg = temp_df[col].rolling(window=WINDOW, min_periods=1).mean().iloc[-1]
        base_features[col_name] = rolling_avg

    # Symptom Rolling Count
    SYMPTOM_COLS = ['PimplesYN', 'hairgrowthYN']
    for col in SYMPTOM_COLS:
        col_name = f'{col}_Count_{WINDOW}d'
        rolling_sum = temp_df[col].rolling(window=WINDOW, min_periods=1).sum().iloc[-1]
        base_features[col_name] = rolling_sum


    # 3. Final Prediction Vector Preparation
    base_features['WeightKg'] = new_entry_dict['WeightKg']
    base_features['BMI'] = new_entry_dict['BMI']

    X_new = pd.DataFrame([base_features], columns=X_cols)
    
    # 4. Prediction
    prediction_proba = xgb_model.predict_proba(X_new)[:, 1][0]
    
    # 5. SHAP Calculation
    shap_values = explainer.shap_values(X_new)
    
    return prediction_proba, shap_values, X_new

# --- STREAMLIT FRONTEND ---

st.title("🎯 CollabBoard PCOS Risk Predictor")
st.markdown("---")

# --- STREAK & STATIC DATA SIDEBAR ---
with st.sidebar:
    st.header("Patient Static Data")
    
    # Display the static features that were loaded
    st.metric("Age (yrs)", st.session_state['base_features']['Ageyrs'])
    st.metric("Height (Cm)", st.session_state['base_features']['HeightCm'])

    # Display the Streak
    today = datetime.date.today()
    
    # Streak logic is calculated and updated if the user submits a new entry on a new day
    if st.session_state['last_entry_date'] == today:
        status_text = "Already logged today!"
    elif st.session_state['last_entry_date'] == today - datetime.timedelta(days=1):
        # We only update the streak state if a submission happens on the next day, 
        # but we don't increment here, we let the submission handler do it.
        status_text = "Ready to continue streak."
    elif st.session_state['last_entry_date'] < today - datetime.timedelta(days=1):
        status_text = "Streak will reset on next log."
    else:
        status_text = "Ready to log today."

    st.subheader("Tracking Status")
    st.metric(label="Current Log Streak", value=f"{st.session_state['streak']} Days 🔥")
    st.caption(f"Last Log Date: {st.session_state['last_entry_date']}")


# --- DAILY INPUT FORM ---
with st.form("daily_entry_form"):
    st.subheader("📥 Today's Health Log")
    col1, col2, col3 = st.columns(3)
    
    # Using last entry value as default for a smooth UX
    with col1:
        weight = st.number_input("Weight (Kg)", value=st.session_state['history'][-1]['WeightKg'], min_value=30.0, step=0.1)
        fast_food = st.checkbox("Ate Fast Food Today?", value=st.session_state['history'][-1]['FastfoodYN'])
    
    with col2:
        reg_exercise = st.checkbox("Regular Exercise Done?", value=st.session_state['history'][-1]['RegExerciseYN'])
        skindarkening = st.checkbox("Skin Darkening Noticed?", value=st.session_state['history'][-1]['SkindarkeningYN'])
    
    with col3:
        pimples = st.checkbox("Pimples Present?", value=st.session_state['history'][-1]['PimplesYN'])
        hair_growth = st.checkbox("Hair Growth Increase?", value=st.session_state['history'][-1]['hairgrowthYN'])
        hair_loss = st.checkbox("Hair Loss Noticed?", value=st.session_state['history'][-1]['HairlossYN'])

    submitted = st.form_submit_button("Submit Daily Log & Predict")


# --- PREDICTION & SHAP OUTPUT ---
if submitted:
    
    # 1. Update Streak State before prediction (only if it's a new day)
    current_date = datetime.date.today()
    last_log_date = st.session_state['last_entry_date']
    
    if current_date > last_log_date:
        if current_date == last_log_date + datetime.timedelta(days=1):
            st.session_state['streak'] += 1
        else:
            st.session_state['streak'] = 1 # Streak broken
        st.session_state['last_entry_date'] = current_date
        # st.experimental_rerun() is REMOVED here to prevent double execution if the
        # main content (prediction) is visible. Streak update will be reflected on the next action.
        
    
    # 2. Create the new entry dictionary
    new_entry = {
        'Date': current_date,
        'PatientID': st.session_state['history'][-1]['PatientID'], 
        'WeightKg': weight,
        'BMI': weight / ((st.session_state['base_features']['HeightCm'] / 100)**2), # Calculate current BMI
        'PimplesYN': int(pimples),
        'hairgrowthYN': int(hair_growth),
        'SkindarkeningYN': int(skindarkening),
        'HairlossYN': int(hair_loss),
        'FastfoodYN': int(fast_food),
        'RegExerciseYN': int(reg_exercise),
    }

    # 3. Get Prediction and SHAP values
    prediction_score, shap_values, X_new = prepare_and_predict(
        new_entry, 
        st.session_state['base_features'].copy(), 
        st.session_state['history'], 
        X_train_cols
    )

    # 4. Update History (Crucial step)
    # We update history AFTER prediction, so the next run includes today's data in the rolling window
    st.session_state['history'].append(new_entry) 
    
    st.markdown("---")
    st.header(f"📈 Prediction for {current_date.strftime('%Y-%m-%d')}")

    # Display Prediction
    risk_level = "HIGH RISK" if prediction_score >= 0.5 else "LOW RISK"
    color = "red" if prediction_score >= 0.5 else "green"

    st.markdown(f"## Predicted PCOS Risk Score: <span style='color:{color};'>**{prediction_score:.2f}**</span>", unsafe_allow_html=True)
    st.write(f"This score suggests a **{risk_level}** of progression or diagnosis, based on your last {WINDOW} days of data.")

    # 5. Generate and Display Natural Language Explanation
    st.subheader("🗣️ AI-Powered Explanation")
    
    # Use a placeholder container to update the status during retries
    explanation_container = st.empty() 
    
    with explanation_container:
        with st.spinner("Generating personalized explanation..."):
            expected_value = explainer.expected_value 
            shap_explanation = generate_shap_explanation(
                X_new, 
                shap_values, 
                expected_value, 
                prediction_score
            )
        explanation_container.markdown(shap_explanation) # Display the final markdown response

    # 6. Display SHAP Explanation (Force Plot)
    st.subheader("🔬 Risk Factor Breakdown (SHAP)")
    st.markdown("The chart below explains **why** the model produced this specific score.")
    
    try:
        # Create a matplotlib figure for rendering
        plt.clf() 
        fig = shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            X_new.iloc[0],
            matplotlib=True,
            show=False
        )
        st.pyplot(fig, bbox_inches='tight')
    except Exception as e:
        st.error(f"Could not render SHAP plot: {e}")
    
    st.caption("Red features increase the risk score; Blue features decrease it.")