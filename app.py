import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np
from pathlib import Path

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================================
# CUSTOM CSS STYLING
# ========================================
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom container styling */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 2rem auto;
        max-width: 1400px;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-in;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Input section styling */
    .input-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    
    /* Result section styling */
    .result-section {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        min-height: 400px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    /* Success/Warning cards */
    .prediction-card {
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
        width: 100%;
    }
    
    .success-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .prediction-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .prediction-text {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .probability-text {
        font-size: 1.3rem;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 3rem;
        font-size: 1.2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: white;
        padding: 2rem;
        font-size: 1rem;
        margin-top: 3rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Section headers */
    .section-header {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Streamlit selectbox styling */
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ========================================
# LOAD MODELS
# ========================================
@st.cache_resource
def load_models():
    """Load artifacts with sanity checks.

    Order of preference:
    1) pipeline.pkl (single fitted Pipeline: preprocessor → LDA → model)
    2) Separate files: preprocessor.pkl, lda.pkl, best_model.pkl
    Also load cols_info.pkl if present.
    """
    base_path = Path(__file__).parent
    pipeline_path = base_path / 'pipeline.pkl'
    preproc_path = base_path / 'preprocessor.pkl'
    lda_path = base_path / 'lda.pkl'
    model_path = base_path / 'best_model.pkl'

    def _load(path, name):
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at {path}")
        try:
            return joblib.load(path)
        except Exception:
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load {name} from {path}: {e}")

    try:
        # Prefer a single pipeline if available
        pipeline = None
        try:
            pipeline = _load(pipeline_path, 'pipeline')
        except FileNotFoundError:
            pipeline = None

        if pipeline is not None:
            # Attempt to introspect embedded steps
            preprocessor = None
            lda = None
            model = None
            try:
                from sklearn.pipeline import Pipeline
                if hasattr(pipeline, 'named_steps'):
                    preprocessor = pipeline.named_steps.get('preprocessor')
                    lda = pipeline.named_steps.get('lda')
                    model = pipeline.named_steps.get('model')
            except Exception:
                pass
            return preprocessor or pipeline, lda, model or pipeline, {}

        # Fall back to separate artifacts
        preprocessor = _load(preproc_path, 'preprocessor')
        lda = _load(lda_path, 'lda')
        model = _load(model_path, 'best_model')

        # Hard sanity check: preprocessor output size must equal LDA's n_features_in_
        try:
            try:
                pre_feat_ct = len(preprocessor.get_feature_names_out())
            except Exception:
                # Rough fallback: transform a dummy frame if cols_info exists
                pre_feat_ct = None
            lda_exp = getattr(lda, 'n_features_in_', None)
            if lda_exp is None and hasattr(lda, 'scalings_') and getattr(lda, 'scalings_', None) is not None:
                lda_exp = lda.scalings_.shape[0]
            if (pre_feat_ct is not None) and (lda_exp is not None) and (pre_feat_ct != lda_exp):
                st.error(
                    f"❌ Inconsistent artifacts: preprocessor produces {pre_feat_ct} features but LDA expects {lda_exp}.\n"
                    "They must come from the same training run. Re-export a single pipeline.pkl or matching preprocessor.pkl + lda.pkl."
                )
                st.stop()
        except Exception:
            pass

        # Load cols_info if present (contains 'num_cols' and 'cat_cols')
        cols_info_path = base_path / 'cols_info.pkl'
        try:
            cols_info = _load(cols_info_path, 'cols_info')
        except FileNotFoundError:
            cols_info = {}

        return preprocessor, lda, model, cols_info
    except FileNotFoundError as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()
    except RuntimeError as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

preprocessor, lda, model, cols_info = load_models()

# ========================================
# HEADER
# ========================================
st.markdown('<h1 class="main-title">💼 Employee Attrition Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict whether an employee is likely to stay or leave using advanced ML models</p>', unsafe_allow_html=True)

# ========================================
# MAIN LAYOUT
# ========================================
col1, col2 = st.columns([1, 1], gap="large")

# ========================================
# LEFT COLUMN - INPUT SECTION
# ========================================
with col1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📋 Employee Information</h3>', unsafe_allow_html=True)

    # Categorical inputs
    department = st.selectbox("🏢 Department", ["Research & Development", "Sales", "Human Resources"])
    business_travel = st.selectbox("✈️ Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    marital_status = st.selectbox("💑 Marital Status", ["Single", "Married", "Divorced"])
    education_field = st.selectbox("🎓 Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
    job_role = st.selectbox("👔 Job Role", [
        "Manager", "Research Scientist", "Laboratory Technician",
        "Sales Executive", "Research Director", "Healthcare Representative",
        "Manufacturing Director", "Human Resources", "Sales Representative"
    ])
    over_time = st.selectbox("⏱️ OverTime", ["Yes", "No"])
    gender = st.selectbox("⚧ Gender", ["Male", "Female"])

    # Numeric inputs
    job_involvement = st.slider("📊 Job Involvement", 1, 4, 3)
    stock_option_level = st.slider("💰 Stock Option Level", 0, 3, 1)
    job_level = st.slider("🏆 Job Level", 1, 5, 2)
    environment_satisfaction = st.slider("🌿 Environment Satisfaction", 1, 4, 3)
    work_life_balance = st.slider("⚖️ Work-Life Balance", 1, 4, 3)
    job_satisfaction = st.slider("😊 Job Satisfaction", 1, 4, 3)
    relationship_satisfaction = st.slider("🤝 Relationship Satisfaction", 1, 4, 3)
    num_companies_worked = st.number_input("🏢 Number of Companies Worked", min_value=0, max_value=10, value=1)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔮 Predict Attrition", use_container_width=True)


# ========================================
# RIGHT COLUMN - RESULTS SECTION
# ========================================
with col2:
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    
    if predict_button:
        try:
            # 1) Determine raw schema used at training (ordered)
            num_cols = cols_info.get('num_cols', [])
            cat_cols = cols_info.get('cat_cols', [])

            # Fallback: infer from fitted ColumnTransformer if cols_info is missing
            if (not num_cols or not cat_cols) and hasattr(preprocessor, 'transformers_'):
                inferred_num, inferred_cat = [], []
                for name, trans, cols in preprocessor.transformers_:
                    if name == 'remainder':
                        continue
                    # Heuristic: OneHotEncoder branch -> categorical; others -> numeric
                    try:
                        from sklearn.preprocessing import OneHotEncoder
                        base_est = trans.steps[-1][1] if hasattr(trans, 'steps') else trans
                        if isinstance(base_est, OneHotEncoder):
                            inferred_cat.extend(list(cols))
                        else:
                            inferred_num.extend(list(cols))
                    except Exception:
                        # If unsure, just append to numeric to keep coverage
                        inferred_num.extend(list(cols))
                if not num_cols:
                    num_cols = inferred_num
                if not cat_cols:
                    cat_cols = inferred_cat

            # 2) Build defaults from the training dataset if present (or simple fallbacks)
            defaults = {}
            try:
                df_defaults = pd.read_csv('employee_cleaned.csv')
                for c in num_cols:
                    defaults[c] = float(df_defaults[c].median()) if c in df_defaults.columns else 0.0
                for c in cat_cols:
                    defaults[c] = df_defaults[c].mode()[0] if c in df_defaults.columns else ''
            except Exception:
                for c in num_cols:
                    defaults[c] = 0.0
                for c in cat_cols:
                    defaults[c] = ''

            # 3) Compose a single raw input row, filling UI where available and defaults elsewhere
            raw_input = {c: defaults.get(c, np.nan) for c in (num_cols + cat_cols)}

            # Fill categorical selections available from UI
            if 'Department' in raw_input:
                raw_input['Department'] = department
            if 'BusinessTravel' in raw_input:
                raw_input['BusinessTravel'] = business_travel
            if 'MaritalStatus' in raw_input:
                raw_input['MaritalStatus'] = marital_status
            if 'EducationField' in raw_input:
                raw_input['EducationField'] = education_field
            if 'JobRole' in raw_input:
                raw_input['JobRole'] = job_role
            if 'OverTime' in raw_input:
                raw_input['OverTime'] = over_time
            if 'Gender' in raw_input:
                raw_input['Gender'] = gender

            # Fill numeric inputs from UI only if the column exists in training schema
            ui_numeric = {
                'JobInvolvement': job_involvement,
                'StockOptionLevel': stock_option_level,
                'JobLevel': job_level,
                'EnvironmentSatisfaction': environment_satisfaction,
                'WorkLifeBalance': work_life_balance,
                'JobSatisfaction': job_satisfaction,
                'RelationshipSatisfaction': relationship_satisfaction,
                'NumCompaniesWorked': num_companies_worked,
            }
            for k, v in ui_numeric.items():
                if k in raw_input:
                    raw_input[k] = v

            # 4) Create DataFrame in exact training order and cast dtypes
            raw_df = pd.DataFrame([raw_input], columns=(num_cols + cat_cols))
            # Cast numerics to float and categoricals to string/object to match encoder expectations
            for c in num_cols:
                if c in raw_df.columns:
                    raw_df[c] = pd.to_numeric(raw_df[c], errors='coerce').astype(float)
            for c in cat_cols:
                if c in raw_df.columns:
                    raw_df[c] = raw_df[c].astype(str)

            # 5) Sanity check: preprocessor output size vs LDA expectation
            expected_features = None
            if hasattr(lda, 'scalings_') and getattr(lda, 'scalings_') is not None:
                try:
                    expected_features = lda.scalings_.shape[0]
                except Exception:
                    expected_features = None
            if expected_features is None:
                expected_features = getattr(lda, 'n_features_in_', None)

            X_proc = preprocessor.transform(raw_df)
            proc_features = X_proc.shape[1]

            if (expected_features is not None) and (proc_features != expected_features):
                # Provide concise guidance and stop to avoid wrong predictions
                st.error(f"❌ Feature count mismatch: {proc_features} vs {expected_features}")
                st.info("💡 Fix: Use the same raw schema as training; save raw_schema + defaults during training and load them here.")
                raise ValueError("Feature count mismatch")

            # 6) Predict: LDA → model
            X_lda_input = lda.transform(X_proc)
            prediction = model.predict(X_lda_input)
            prediction_proba = model.predict_proba(X_lda_input)

            confidence = prediction_proba[0][prediction[0]] * 100

            if prediction[0] == 0:  # Employee stays
                st.markdown(f"""
                    <div class=\"prediction-card success-card\">\n                        <div class=\"prediction-icon\">✅</div>\n                        <div class=\"prediction-text\">Employee is Likely to Stay</div>\n                        <div class=\"probability-text\">Confidence: {confidence:.1f}%</div>\n                    </div>
                """, unsafe_allow_html=True)
            else:  # Employee leaves
                st.markdown(f"""
                    <div class=\"prediction-card warning-card\">\n                        <div class=\"prediction-icon\">⚠️</div>\n                        <div class=\"prediction-text\">Employee is Likely to Leave</div>\n                        <div class=\"probability-text\">Confidence: {confidence:.1f}%</div>\n                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.info("💡 Tip: Always pass the complete raw training schema; this app fills missing fields using dataset defaults and applies the saved preprocessor → LDA → model pipeline.")
    
    else:
        # Default placeholder
        st.markdown("""
            <div style="text-align: center; color: #999; padding: 3rem;">
                <div style="font-size: 5rem; margin-bottom: 1rem;">🔮</div>
                <div style="font-size: 1.5rem; font-weight: 600;">Ready to Predict</div>
                <div style="font-size: 1rem; margin-top: 0.5rem;">Fill in the employee details and click the predict button</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ========================================
# FOOTER
# ========================================
st.markdown("""
    <div class="footer">
        Built with ❤️ using Streamlit and Scikit-learn
    </div>
""", unsafe_allow_html=True)