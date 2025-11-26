import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import datetime
from collections import deque
import xgboost as xgb

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

# --- app.py: UPDATED load_assets function ---

@st.cache_resource
def load_assets():
    """Loads the model, explainer, and feature list once."""
    try:
        # 1. Load the trained model using the native XGBoost method
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model('streamlit_assets/xgb_model.json')
        
        # 2. Load the list of features used in training
        X_train_cols = joblib.load('streamlit_assets/feature_columns.joblib')
        
        # 3. Initialize the SHAP Explainer from the loaded model
        explainer = shap.TreeExplainer(xgb_model)
        
        # 4. Load the base data
        base_df = pd.read_csv('streamlit_assets/base_patient_data.csv')

        return xgb_model, explainer, X_train_cols, base_df
    
    except FileNotFoundError as e:
        st.error(f"Error loading required files: {e}. "
                 "Ensure all assets are in the 'streamlit_assets' directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during loading: {e}")
        st.stop()


@st.cache_data
def get_initial_history(base_df, patient_id=1):
    """Gets the history for a single patient from the base data."""
    # We use a single patient's data as a template for the history
    patient_data = base_df[base_df['PatientID'] == patient_id].iloc[0].to_dict()

    # Create a deque (double-ended queue) for fast history management
    history = deque(maxlen=WINDOW)

    # Populate the history with the patient's initial state for the full window
    for day in range(WINDOW):
        # Create an initial entry, potentially with some noise for realism
        entry = {
            'Date': datetime.date.today() - datetime.timedelta(days=WINDOW - 1 - day),
            'PatientID': patient_id
        }
        for feature in DAILY_FEATURES:
            # Use the base value for the history
            entry[feature] = patient_data[feature]
        
        # Static features are not needed in the history deque, only in the final prediction vector

        history.append(entry)
    
    return history, patient_data


# --- SESSION STATE INITIALIZATION ---
# Load all assets
xgb_model, explainer, X_train_cols, base_df = load_assets()

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
    
    # 2. Calculate Rolling Features (Lagging is built into the rolling window logic)
    # We calculate the rolling features on the last row of the combined dataframe
    
    # Continuous Rolling Average
    ROLLING_COLS = ['WeightKg', 'BMI']
    for col in ROLLING_COLS:
        col_name = f'{col}_Avg_{WINDOW}d'
        # Calculates the mean of the *previous* WINDOW entries up to the current entry.
        # This is the value that will be used for the current day's prediction.
        rolling_avg = temp_df[col].rolling(window=WINDOW, min_periods=1).mean().iloc[-1]
        base_features[col_name] = rolling_avg

    # Symptom Rolling Count
    SYMPTOM_COLS = ['PimplesYN', 'hairgrowthYN']
    for col in SYMPTOM_COLS:
        col_name = f'{col}_Count_{WINDOW}d'
        # Calculates the sum of the *previous* WINDOW entries up to the current entry.
        rolling_sum = temp_df[col].rolling(window=WINDOW, min_periods=1).sum().iloc[-1]
        base_features[col_name] = rolling_sum


    # 3. Final Prediction Vector Preparation
    # Get the current daily values for Weight and BMI
    base_features['WeightKg'] = new_entry_dict['WeightKg']
    base_features['BMI'] = new_entry_dict['BMI']

    # Filter for the exact columns used in training and convert to DataFrame
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
    if st.session_state['last_entry_date'] == today:
        status_text = "Already logged today!"
    elif st.session_state['last_entry_date'] == today - datetime.timedelta(days=1):
        st.session_state['streak'] += 1
        st.session_state['last_entry_date'] = today
        status_text = "Streak updated!"
    elif st.session_state['last_entry_date'] < today - datetime.timedelta(days=1):
        st.session_state['streak'] = 1 # Reset if missed a day
        st.session_state['last_entry_date'] = today
        status_text = "New streak started!"
    else:
        status_text = "Ready to log today."

    st.subheader("Tracking Status")
    st.metric(label="Current Log Streak", value=f"{st.session_state['streak']} Days 🔥")
    st.caption(f"Last Log: {st.session_state['last_entry_date']}")


# --- DAILY INPUT FORM ---
with st.form("daily_entry_form"):
    st.subheader("📥 Today's Health Log")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        weight = st.number_input("Weight (Kg)", value=st.session_state['history'][-1]['WeightKg'], min_value=30.0, step=0.1)
        fast_food = st.checkbox("Ate Fast Food Today?", value=st.session_state['history'][-1]['FastfoodYN'])
    
    with col2:
        # BMI is calculated, not directly entered, but we use the input field for BMI as a proxy for the last recorded value
        # In a real app, BMI should be calculated from Weight/Height, but for this demo, we'll let the model calculate it.
        reg_exercise = st.checkbox("Regular Exercise Done?", value=st.session_state['history'][-1]['RegExerciseYN'])
        skindarkening = st.checkbox("Skin Darkening Noticed?", value=st.session_state['history'][-1]['SkindarkeningYN'])
    
    with col3:
        pimples = st.checkbox("Pimples Present?", value=st.session_state['history'][-1]['PimplesYN'])
        hair_growth = st.checkbox("Hair Growth Increase?", value=st.session_state['history'][-1]['hairgrowthYN'])
        hair_loss = st.checkbox("Hair Loss Noticed?", value=st.session_state['history'][-1]['HairlossYN'])

    submitted = st.form_submit_button("Submit Daily Log & Predict")


# --- PREDICTION & SHAP OUTPUT ---
if submitted:
    
    # 1. Create the new entry dictionary
    new_entry = {
        'Date': datetime.date.today(),
        'PatientID': st.session_state['history'][-1]['PatientID'], # Keep the same ID
        'WeightKg': weight,
        'BMI': weight / ((st.session_state['base_features']['HeightCm'] / 100)**2), # Calculate current BMI
        'PimplesYN': int(pimples),
        'hairgrowthYN': int(hair_growth),
        'SkindarkeningYN': int(skindarkening),
        'HairlossYN': int(hair_loss),
        'FastfoodYN': int(fast_food),
        'RegExerciseYN': int(reg_exercise),
    }

    # 2. Get Prediction and SHAP values
    prediction_score, shap_values, X_new = prepare_and_predict(
        new_entry, 
        st.session_state['base_features'].copy(), # Pass a copy to avoid modification
        st.session_state['history'], 
        X_train_cols
    )

    # 3. Update History (Crucial step)
    st.session_state['history'].append(new_entry)
    
    st.markdown("---")
    st.header(f"📈 Prediction for {new_entry['Date'].strftime('%Y-%m-%d')}")

    # Display Prediction
    risk_level = "HIGH RISK" if prediction_score >= 0.5 else "LOW RISK"
    color = "red" if prediction_score >= 0.5 else "green"

    st.markdown(f"## Predicted PCOS Risk Score: <span style='color:{color};'>**{prediction_score:.2f}**</span>", unsafe_allow_html=True)
    st.write(f"This score suggests a **{risk_level}** of progression or diagnosis, based on your last {WINDOW} days of data.")

    # 4. Display SHAP Explanation (Force Plot)
    st.subheader("🔬 Risk Factor Breakdown (SHAP)")
    st.markdown("The chart below explains **why** the model produced this specific score.")
    
    # SHAP force plot requires matplotlib=True for Streamlit to render it via st.pyplot()
    try:
        # Create a matplotlib figure for rendering
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