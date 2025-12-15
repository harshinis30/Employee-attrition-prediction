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
# ENHANCED CUSTOM CSS STYLING
# ========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism container */
    .main-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 30px;
        padding: 3rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 2rem auto;
        max-width: 1600px;
    }
    
    /* Animated title with gradient text */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-in, pulse 2s ease-in-out infinite;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.3);
        letter-spacing: 2px;
        background: none !important;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.3rem;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 1px;
        animation: fadeIn 1.5s ease-in;
    }
    
    /* Enhanced input section with glassmorphism */
    .input-section {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .input-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Enhanced result section */
    .result-section {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        min-height: 500px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: transform 0.3s ease;
    }
    
    .result-section:hover {
        transform: translateY(-5px);
    }
    
    /* Enhanced prediction cards with animations */
    .prediction-card {
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        animation: slideIn 0.6s ease-out, float 3s ease-in-out infinite;
        width: 100%;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(30px) scale(0.9); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
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
        font-size: 5rem;
        margin-bottom: 1.5rem;
        animation: bounce 1s ease-in-out infinite;
        filter: drop-shadow(0 5px 15px rgba(0, 0, 0, 0.3));
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-15px); }
    }
    
    .prediction-text {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        letter-spacing: 1px;
    }
    
    .probability-text {
        font-size: 1.5rem;
        opacity: 0.95;
        font-weight: 600;
        text-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced button with gradient and animations */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-size: 1.3rem;
        font-weight: 700;
        width: 100%;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.7);
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Enhanced footer */
    .footer {
        text-align: center;
        color: white;
        padding: 2rem;
        font-size: 1.1rem;
        margin-top: 3rem;
        font-weight: 300;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        animation: fadeIn 2s ease-in;
    }
    
    .footer-heart {
        color: #ff6b6b;
        animation: heartbeat 1.5s ease-in-out infinite;
        display: inline-block;
    }
    
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.2); }
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Enhanced section headers */
    .section-header {
        color: #1a1a1a !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-align: center;
        letter-spacing: 1px;
    }
    
    /* Enhanced input styling */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div > div > div {
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Label styling */
    .stSelectbox label,
    .stNumberInput label,
    .stSlider label {
        font-weight: 600;
        color: #333;
        font-size: 1rem;
    }
    
    /* Placeholder styling */
    .placeholder-container {
        text-align: center;
        color: #999;
        padding: 3rem;
        animation: fadeIn 1s ease-in;
    }
    
    .placeholder-icon {
        font-size: 6rem;
        margin-bottom: 1.5rem;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 5px 15px rgba(0, 0, 0, 0.1));
    }
    
    .placeholder-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .placeholder-subtitle {
        font-size: 1.1rem;
        color: #999;
    }
    
    /* Hide Streamlit branding and empty containers */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Remove Streamlit's default padding and empty containers */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
    }
            
        /* Remove extra white boxes / containers */
    div[data-testid="stHorizontalBlock"] > div:first-child:empty,
    div[data-testid="stHorizontalBlock"] > div:nth-child(2):empty {
        display: none !important;
    }

    div[data-testid="stVerticalBlock"] > div:empty,
    div[data-testid="stVerticalBlock"] > div > div:empty {
        display: none !important;
    }

    /* Remove background color on empty column blocks */
    div[data-testid="column"]:has(> div:empty) {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    
    /* Hide empty div containers that Streamlit creates */
    div[data-testid="stVerticalBlock"] > div:empty {
        display: none;
    }
    
    div[data-testid="column"] > div:empty {
        display: none;
    }
    
    /* Remove extra spacing from Streamlit columns */
    div[data-testid="column"] {
        background: transparent !important;
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
            
    /* ===== Dark Text Theme for Better Readability ===== */

    /* Main title */
    h1, h2, h3, .headline, .dashboard-title {
        color: #1b1b1b !important;            /* near-black */
        text-shadow: 0px 1px 3px rgba(255,255,255,0.2); /* subtle lift */
    }

    /* Subtitle (under main title) */
    .subtitle, .subheadline, p {
        color: #2e2e2e !important;            /* medium-dark gray */
    }

    /* Section headers (like "Employee Information") */
    .section-title, h4, h5, h6 {
        color: #242424 !important;            /* slightly softer black */
        font-weight: 700 !important;
    }

    /* "Ready to Predict" and its description */
    .placeholder-title {
        color: #1e1e1e !important;
        text-shadow: 0 1px 2px rgba(255, 255, 255, 0.3);
    }

    .placeholder-subtitle {
        color: #3b3b3b !important;
        opacity: 0.9;
    }
            
    /* Force dark colors on Streamlit headings */

    /* Main title ("Employee Attrition Prediction Dashboard") */
    section.main > div:first-child h1, h1, .stMarkdown h1 {
        color: #1a1a1a !important;           /* dark gray-black */
        text-shadow: none !important;
    }

    /* Subtitle (the line below main title) */
    section.main > div:first-child p, .stMarkdown p {
        color: #2c2c2c !important;
    }

    /* Section header ("Employee Information") */
    div[data-testid="stHeading"], h2, h3, .stMarkdown h2, .stMarkdown h3 {
        color: #202020 !important;           /* strong dark */
        font-weight: 700 !important;
        text-shadow: none !important;
    }

    /* "Ready to Predict" block */
    .placeholder-title {
        color: #121212 !important;

            /* Main title ("Employee Attrition Prediction Dashboard") */
        section.main > div:first-child h1, h1, .stMarkdown h1 {
            color: #1a1a1a !important;           /* dark gray-black */
            text-shadow: none !important;
        }

        /* Section header ("Employee Information") */
        div[data-testid="stHeading"], h2, h3, .stMarkdown h2, .stMarkdown h3 {
            color: #202020 !important;           /* strong dark */
            font-weight: 700 !important;
            text-shadow: none !important;
        }
            
        /* Force main title to be dark - override all other rules */
        .main-title {
            color: #1a1a1a !important;
            background: none !important;
            -webkit-text-fill-color: #1a1a1a !important;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }

    </style>
""", unsafe_allow_html=True)

# ========================================
# LOAD MODELS
# ========================================
@st.cache_resource
def load_models():
    """Load artifacts with sanity checks."""
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

        # Sanity check
        try:
            try:
                pre_feat_ct = len(preprocessor.get_feature_names_out())
            except Exception:
                pre_feat_ct = None
            lda_exp = getattr(lda, 'n_features_in_', None)
            if lda_exp is None and hasattr(lda, 'scalings_') and getattr(lda, 'scalings_', None) is not None:
                lda_exp = lda.scalings_.shape[0]
            if (pre_feat_ct is not None) and (lda_exp is not None) and (pre_feat_ct != lda_exp):
                st.error(
                    f"❌ Inconsistent artifacts: preprocessor produces {pre_feat_ct} features but LDA expects {lda_exp}."
                )
                st.stop()
        except Exception:
            pass

        # Load cols_info if present
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
st.markdown('<p class="subtitle">✨ Predict whether an employee is likely to stay or leave using advanced ML models ✨</p>', unsafe_allow_html=True)

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

    # Categorical inputs with enhanced styling
    department = st.selectbox(
        "🏢 Department",
        ["Research & Development", "Sales", "Human Resources"],
        help="Select the employee's department"
    )
    
    business_travel = st.selectbox(
        "✈️ Business Travel",
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        help="How often does the employee travel for business?"
    )
    
    marital_status = st.selectbox(
        "💑 Marital Status",
        ["Single", "Married", "Divorced"],
        help="Employee's current marital status"
    )
    
    education_field = st.selectbox(
        "🎓 Education Field",
        ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
        help="Employee's field of education"
    )
    
    job_role = st.selectbox(
        "👔 Job Role",
        [
            "Manager", "Research Scientist", "Laboratory Technician",
            "Sales Executive", "Research Director", "Healthcare Representative",
            "Manufacturing Director", "Human Resources", "Sales Representative"
        ],
        help="Employee's current job role"
    )
    
    over_time = st.selectbox(
        "⏱️ OverTime",
        ["Yes", "No"],
        help="Does the employee work overtime?"
    )
    
    gender = st.selectbox(
        "⚧ Gender",
        ["Male", "Female"],
        help="Employee's gender"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Numeric inputs with sliders
    job_involvement = st.slider(
        "📊 Job Involvement",
        1, 4, 3,
        help="Level of job involvement (1-4)"
    )
    
    stock_option_level = st.slider(
        "💰 Stock Option Level",
        0, 3, 1,
        help="Stock option level (0-3)"
    )
    
    job_level = st.slider(
        "🏆 Job Level",
        1, 5, 2,
        help="Current job level (1-5)"
    )
    
    environment_satisfaction = st.slider(
        "🌿 Environment Satisfaction",
        1, 4, 3,
        help="Satisfaction with work environment (1-4)"
    )
    
    work_life_balance = st.slider(
        "⚖️ Work-Life Balance",
        1, 4, 3,
        help="Work-life balance rating (1-4)"
    )
    
    job_satisfaction = st.slider(
        "😊 Job Satisfaction",
        1, 4, 3,
        help="Overall job satisfaction (1-4)"
    )
    
    relationship_satisfaction = st.slider(
        "🤝 Relationship Satisfaction",
        1, 4, 3,
        help="Satisfaction with workplace relationships (1-4)"
    )
    
    num_companies_worked = st.number_input(
        "🏢 Number of Companies Worked",
        min_value=0, max_value=10, value=1,
        help="Total number of companies worked at"
    )

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
            with st.spinner('🔮 Analyzing employee data...'):
                # Determine raw schema
                num_cols = cols_info.get('num_cols', [])
                cat_cols = cols_info.get('cat_cols', [])

                # Fallback: infer from fitted ColumnTransformer
                if (not num_cols or not cat_cols) and hasattr(preprocessor, 'transformers_'):
                    inferred_num, inferred_cat = [], []
                    for name, trans, cols in preprocessor.transformers_:
                        if name == 'remainder':
                            continue
                        try:
                            from sklearn.preprocessing import OneHotEncoder
                            base_est = trans.steps[-1][1] if hasattr(trans, 'steps') else trans
                            if isinstance(base_est, OneHotEncoder):
                                inferred_cat.extend(list(cols))
                            else:
                                inferred_num.extend(list(cols))
                        except Exception:
                            inferred_num.extend(list(cols))
                    if not num_cols:
                        num_cols = inferred_num
                    if not cat_cols:
                        cat_cols = inferred_cat

                # Build defaults
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

                # Compose raw input
                raw_input = {c: defaults.get(c, np.nan) for c in (num_cols + cat_cols)}

                # Fill categorical selections
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

                # Fill numeric inputs
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

                # Create DataFrame
                raw_df = pd.DataFrame([raw_input], columns=(num_cols + cat_cols))
                
                # Cast dtypes
                for c in num_cols:
                    if c in raw_df.columns:
                        raw_df[c] = pd.to_numeric(raw_df[c], errors='coerce').astype(float)
                for c in cat_cols:
                    if c in raw_df.columns:
                        raw_df[c] = raw_df[c].astype(str)

                # Sanity check
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
                    st.error(f"❌ Feature count mismatch: {proc_features} vs {expected_features}")
                    st.info("💡 Fix: Use the same raw schema as training")
                    raise ValueError("Feature count mismatch")

                # Predict
                X_lda_input = lda.transform(X_proc)
                prediction = model.predict(X_lda_input)
                prediction_proba = model.predict_proba(X_lda_input)

                confidence = prediction_proba[0][prediction[0]] * 100

            # Display results with enhanced animations
            if prediction[0] == 0:  # Employee stays
                st.markdown(f"""
                    <div class="prediction-card success-card">
                        <div class="prediction-icon">✅</div>
                        <div class="prediction-text">Employee is Likely to Stay</div>
                        <div class="probability-text">Confidence: {confidence:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            else:  # Employee leaves
                st.markdown(f"""
                    <div class="prediction-card warning-card">
                        <div class="prediction-icon">⚠️</div>
                        <div class="prediction-text">Employee is Likely to Leave</div>
                        <div class="probability-text">Confidence: {confidence:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.info("💡 Tip: Ensure all model files are properly configured")
    
    else:
        # Enhanced placeholder
        st.markdown("""
            <div class="placeholder-container">
                <div class="placeholder-icon">🔮</div>
                <div class="placeholder-title">Ready to Predict</div>
                <div class="placeholder-subtitle">
                    Fill in the employee details and click the predict button
                </div>
            </div>

            <style>
                /* Remove unwanted white boxes / blank divs */
                div[data-testid="stVerticalBlock"] div:empty,
                div[data-testid="stHorizontalBlock"] div:empty,
                div[data-testid="column"] div:empty,
                div:empty {
                    display: none !important;
                    visibility: hidden !important;
                    background: transparent !important;
                    border: none !important;
                    box-shadow: none !important;
                    padding: 0 !important;
                    margin: 0 !important;
                }
            </style>
        """, unsafe_allow_html=True)

# ========================================
# FOOTER
# ========================================
st.markdown("""
    <div class="footer">
        Built with <span class="footer-heart">❤️</span> using Streamlit and Scikit-learn
    </div>
""", unsafe_allow_html=True)