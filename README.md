# Pulse Healthcare AI - Insurance Premium Predictor

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FFCC00?style=for-the-badge&logo=xgboost)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-00B3E6?style=for-the-badge)

Welcome to the **Pulse Healthcare AI** repository. This project is a state-of-the-art machine learning application designed to predict health insurance premiums with exceptionally high accuracy. It serves as an end-to-end demonstration of integrating advanced predictive modeling, robust data engineering, and a highly responsive, custom-built frontend web interface.

---

## Live Application Link

**[Link to Live Streamlit Application]** *(Insert your Streamlit Community Cloud or hosting link here)*

---

## Project Overview & Key Features

The primary objective of this system is to accurately estimate annual healthcare premiums based on 11 critical user parameters, including age, BMI, medical history, region, and lifestyle factors.

- **Custom Frontend Integration:** Unlike standard Streamlit applications, this project utilizes a custom HTML/CSS/JS frontend injected into Streamlit to allow for hardware-accelerated 3D animations, zero-latency state management, and a highly polished corporate aesthetic.
- **Explainable AI (XAI) Dashboards:** Features SHAP-inspired Waterfall charts that break down the exact mathematical contributors to the final predicted premium, ensuring transparency for the end-user.
- **Robust Input Validation:** Implements strict client-side constraints (e.g., rigid min/max boundaries on age and income) to prevent out-of-distribution errors from being sent to the model.

---

## Machine Learning Architecture: Working of the Model

The prediction engine relies on a highly structured, multi-stage pipeline designed to ingest raw user data and transform it into a numerical feature matrix. 

### 1. Data Transformation Pipeline
When a user submits their profile, the data undergoes rigorous real-time transformations that perfectly mirror the training environment:
- **Logarithmic Scaling:** Applied to skewed numerical distributions, such as income, to normalize variance.
- **Target-Mean Encoding:** Categorical variables that carry high cardinality or non-ordinal relationships (such as specific Medical Histories or BMI Categories) are mapped to their historical impact on premium costs.
- **Feature Engineering:** We generate synthesized interaction features. For example, a `health_risk_score` is computed by calculating the compound effect of BMI, Medical History, and Smoking Status.
- **Variance Inflation Factor (VIF) Pruning:** Columns exhibiting high multicollinearity are dynamically dropped prior to inference to maintain model stability.
- **Standardization:** The final feature matrix is scaled using a saved `StandardScaler` to ensure uniform gradient descent compatibility across our tree-based base models.

### 2. Ensemble Stacking Regressor
Instead of relying on a single algorithm, Pulse AI utilizes a powerful **Stacking Architecture**.
- **Base Level Models:** The pipeline feeds the processed 16-feature vector into three distinct, highly tuned models:
  - **XGBoost Regressor** (Optimized for depth and learning rate)
  - **LightGBM Regressor** (Optimized for leaf-wise gradient boosting speed)
  - **Random Forest Regressor** (Optimized for robust variance reduction)
- **Meta-Estimator Level:** The predictions from these three base models are concatenated and passed into a final **Ridge Regression** model. The Ridge regressor applies L2 regularization to optimally weigh the inputs from the base models, producing a final prediction that minimizes error significantly better than any single model could.

---

## Model Accuracy & Evaluation Metrics

The models were extensively cross-validated using a 5-fold KFold split during the training phase. Performance optimization was conducted via RandomizedSearchCV, prioritizing the minimization of Mean Absolute Error (MAE).

### Cross-Validated MAE (Lower is Better)
- **LightGBM:** 781.03
- **Random Forest:** 779.93
- **XGBoost:** 776.20
- **Stacked Meta-Model (Ridge): 771.76**

The final testing phase on unseen data confirmed the superiority of the Stacking architecture:
- **Final Test MAE:** 766.24
- **Final Test RMSE:** 1,139.52
- **Final Test R² Score:** 0.9816 (98.16% Accuracy)

A dedicated "Model Accuracy" dashboard is included within the application to visually communicate these architectural choices and metrics to technical stakeholders.

---

## Repository Structure

```text
Pulse_Healthcare_Premium_Prediction/
 ├── artifacts/                                       # Compiled serialized models and transformers
 │   ├── insurance_model_v1.joblib                    # The final trained Stacking Regressor
 │   ├── preprocessing_schema.joblib                  # Serialized encoding dictionaries
 │   └── scaler.joblib                                # Fitted StandardScaler
 ├── src/
 │   ├── frontend_v8/                                 # Custom HTML/CSS/JS UI components
 │   ├── app.py                                       # Main Streamlit router and dashboard logic
 │   └── predict_pipeline.py                          # Production class for inference and transformation
 ├── Pulse_Healthcare-Insurance-Premium-Prediction.ipynb  # End-to-end model training, EDA, and tuning notebook
 ├── pluse_healthcare_clean.csv                       # Cleaned dataset used for final training
 ├── pulse_healthcare.xlsx                            # Original raw dataset
 ├── requirements.txt                                 # Explicit package dependencies
 └── README.md                                        # Project documentation
```

---

## Local Development & Installation Instructions

To run this application on your local machine, follow these steps:

### 1. Clone the Repository
Open your terminal and clone the repository:
```bash
git clone https://github.com/yourusername/pulse-healthcare-ai.git
cd pulse-healthcare-ai
```

### 2. Set Up a Virtual Environment (Recommended)
It is highly recommended to isolate dependencies using a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the provided requirements file:
```bash
pip install -r requirements.txt
```

### 4. Launch the Application
Run the Streamlit server, pointing it to the main application file inside the `src` directory:
```bash
python -m streamlit run src/app.py
```

### 5. Access the Local Server
Once the server initializes, open your web browser and navigate to:
```text
http://localhost:8501
```

---

## Future Roadmap

- **LLM API Integration ("Talk to Pulse AI"):** Development is underway for a conversational AI agent backend using FastAPI and LangChain. This agent will interview the user via natural language, dynamically extract the 11 required feature parameters, execute the prediction pipeline silently, and explain the AI's premium prediction to the customer in a highly conversational format.
