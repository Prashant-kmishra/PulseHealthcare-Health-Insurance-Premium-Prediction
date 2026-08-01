# Pulse Healthcare Insurance Premium Predictor

An end-to-end Machine Learning pipeline and interactive web application that predicts health insurance premiums.

## Version 2.0 (Segmented Architecture)
The project has been upgraded to a **V2 Segmented Architecture**. During recent data discovery, it was identified that the original monolithic model struggled to accurately predict premiums for younger demographics. To resolve this, the data was segmented and two specialized models were trained:
1. **Youth Model** (`youth_model_v1`): Highly tuned for applicants aged 35 and under.
2. **Senior/Rest Model** (`senior_model_v1`): Highly tuned for applicants over 35.

The backend now uses an intelligent `ModelRouter` to inspect incoming inference requests and dynamically route the data to the correct model behind the scenes. This results in significantly higher accuracy without requiring any changes to the user-facing web app.

## Repository Structure
This repository follows an enterprise-standard layout:
- `data/`: Contains raw datasets and processed/cleaned CSVs used for model training.
- `notebooks/`: Contains the Jupyter notebooks for data exploration and model training (split into V1 Global and V2 Segmented).
- `artifacts/`: Contains the serialized Joblib models, scalers, and encoders.
- `src/`: Contains the Python source code for data pipelines, feature engineering, and inference routing.
- `app/`: Contains the Streamlit web application and static HTML frontend.
- `docs/`: Contains project documentation (BRD, FRD, Model Cards).

## Running the App Locally

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit server from the root directory:
```bash
python -m streamlit run app/pages/app.py
```
