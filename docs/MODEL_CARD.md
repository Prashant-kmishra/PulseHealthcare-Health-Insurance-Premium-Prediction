# Model Card: Pulse Healthcare Insurance Premium Predictor (V2)

## Model Details
- **Architecture**: Stacking Regressor (XGBoost + LightGBM + RandomForest as base learners, Ridge Regression as meta-model).
- **Segmentation**: The V2 architecture employs a routed ensemble of two distinct models:
  - `youth_model_v1`: Trained exclusively on data for individuals aged 25 and under.
  - `senior_model_v1`: Trained exclusively on data for individuals older than 25.
- **Task**: Regression (Predicting Annual Health Insurance Premiums in INR).

## Intended Use
- **Primary Use Case**: Predicting health insurance premiums for potential customers based on demographic and health data.
- **Out of Scope**: Predicting claims payouts, life expectancy, or denial probability.

## Factors
The model considers the following features:
- **Demographics**: Age, Gender, Region, Marital Status, Number of Dependants.
- **Health**: BMI Category, Smoking Status, Medical History.
- **Financial**: Employment Status, Income Level.
- **Product**: Selected Insurance Plan Tier (Bronze, Silver, Gold).

## Data Processing
The inference pipeline applies rigorous feature engineering mirroring the training setup:
- One-Hot Encoding for categorical demographic variables (drop_first=True).
- Ordinal Encoding for ordinal variables (Insurance Plan, Employment).
- Target-Mean Encoding for high-cardinality categorical variables.
- Dynamic calculated features (Income per capita, engineered risk scores).
- Standardization (StandardScaler).

## Evaluation Data & Metrics
During V1 discovery, it was identified that the monolithic model exhibited poor accuracy and high variance for the youth demographic (age <= 25) because the feature distributions for older demographics were overpowering the gradients during training. 

By segmenting the dataset into two mutually exclusive cohorts and training separate stacking regressors, the V2 architecture drastically reduced Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for the youth cohort without sacrificing accuracy on the senior cohort.

## Ethical Considerations & Caveats
- **Risk Avoidance**: The model should not be used in a way that discriminates protected classes. Features like gender and marital status are included strictly for actuarial calculation mirroring historical cost data.
- **Missing Categories**: If an unseen category is encountered in production, the Target-Mean encoders will safely fall back to the global mean premium observed during training.
