# Employee Attrition Prediction

A machine learning project that analyzes factors influencing employee attrition and predicts attrition risk using advanced classification models and Linear Discriminant Analysis (LDA).

## 📊 Project Overview

This project develops a predictive model to identify key drivers of employee turnover, enabling organizations to make data-driven HR decisions and reduce recruitment costs. The solution includes an interactive Streamlit web application for real-time attrition risk prediction.


## 🎯 Objectives

- Analyze factors influencing employee attrition
- Understand key drivers of employee turnover
- Develop predictive models to identify attrition risk factors
- Support data-driven HR decision-making
- Improve retention strategies and reduce recruitment costs

## 📁 Dataset

- **Source**: IBM HR Analytics Employee Attrition & Performance dataset (Kaggle)
- **Size**: 1470+ employee records
- **Features**: 35 attributes (mix of numerical and categorical variables)
- **Target**: Employee attrition status

## 🔄 Project Pipeline

### 1. Data Collection
- Gathered structured HR data from IBM HR Analytics dataset
- Comprehensive employee information including demographics, job roles, and satisfaction metrics

### 2. Data Cleaning & Preprocessing
- ✅ Checked for missing values (none found)
- 🗑️ Removed constant columns: `EmployeeCount`, `Over18`, `StandardHours`
- 📊 Handled outliers using IQR method
- 🔢 Standardized numeric features using Z-scale normalization
- 🏷️ Encoded categorical variables
- ⚙️ Applied binning/discretization for feature engineering

### 3. Exploratory Data Analysis
- Conducted univariate analysis of all features
- Analyzed feature distributions and patterns
- Identified key demographic patterns through visualizations
- Created histograms and categorical plots

### 4. Data Analysis
- **Bivariate Analysis**: Examined relationships between features and attrition
- **Correlation Analysis**: Analyzed correlations between numeric features
- **Statistical Testing**: Performed hypothesis testing (t-test, z-test)
- Generated correlation heatmaps and pairplots

### 5. Feature Engineering
- Created predictive variables from raw data
- Applied feature transformation techniques
- Selected top features based on importance

### 6. Dimensionality Reduction
- Applied **Linear Discriminant Analysis (LDA)** for dimensionality reduction
- Transformed features to maximize class separability
- Reduced feature space while preserving discriminative information

### 7. Model Development
Trained and evaluated multiple classification models:
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

### 8. Model Evaluation
- Compared performance across all models
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Selected best performing model for deployment

### 9. Deployment
- Developed interactive **Streamlit web application**
- Takes top 15 LDA-transformed features as input
- Applies same preprocessing pipeline as training data
- Provides real-time attrition predictions

## 🚀 User Interaction Workflow

1. **User Input**: HR analyst enters employee parameters via Streamlit dashboard
2. **Data Preprocessing**: Input data is standardized and cleaned
3. **Feature Engineering**: Dynamic calculation of necessary features
4. **LDA Transformation**: Features transformed using trained LDA model
5. **Model Prediction**: ML model generates attrition risk score/classification
6. **Result Display**: Predictions and visualizations presented in Streamlit

## 📦 Repository Structure

```
Employee-attrition-prediction/
├── IDSC_PROJECT.ipynb          # Main analysis and modeling notebook
├── app.py                       # Primary Streamlit application
├── app1.py                      # Alternative Streamlit app version
├── dataset1.csv                 # Original dataset
├── employee_cleaned.csv         # Cleaned dataset
├── best_model.pkl              # Trained model (serialized)
├── lda.pkl                     # LDA transformer (serialized)
├── preprocessor.pkl            # Data preprocessor (serialized)
├── feature_info.pkl            # Feature metadata
├── cols_info.pkl               # Column information
├── top_features.pkl            # Top feature list (serialized)
├── top_features.csv            # Top feature list (CSV)
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version specification
└── IDSC_PPT.pdf               # Project presentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.x

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/harshinis30/Employee-attrition-prediction.git
cd Employee-attrition-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit application**
```bash
streamlit run app.py
```

## 💻 Usage

1. Launch the Streamlit app
2. Enter employee information in the interactive dashboard
3. Submit the form to get attrition risk prediction
4. View prediction results and supporting visualizations

## 🔮 Future Work

### Planned Enhancements
- **Real-Time Prediction**: Implement continuous, event-driven attrition prediction
- **Broader Metric Inclusion**: Integrate performance reviews and engagement scores
- **Recommendation System**: Add personalized retention strategies for at-risk employees
- **Interactive HR Dashboard**: Create comprehensive dashboard for deeper operational insights
- **Model Refinement**: Continuous model improvement with new data

## 📊 Key Findings

The analysis focuses on understanding patterns across:
- Demographics (age, gender, marital status)
- Job characteristics (role, department, level)
- Satisfaction levels (job satisfaction, work-life balance)
- Compensation and benefits
- Work environment factors

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is available for educational and research purposes.


**Note**: This project was developed as part of the IDSC (Introduction to Data Science) course project.
