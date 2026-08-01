"""
pipeline.py
-----------
Inference pipeline for the Pulse Healthcare Insurance Premium model.

This mirrors, step-by-step, everything that happened to the data inside
`Pulse_Healthcare-Insurance-Premium-Prediction.ipynb` between raw input and
the exact 16-column matrix the stacked model (XGBoost + LightGBM + RandomForest
-> Ridge meta-model) was trained on:

    1. Column name normalisation (spaces -> underscores, lowercase)
    2. Cleaning (abs() on number_of_dependants, smoking_status label fix)
    3. Feature creation (income_log, per_capita_income, age brackets)
    4. One-Hot Encoding (gender, region -- drop_first=True)
    5. Ordinal Encoding (insurance_plan, employment_status)
    6. Target-Mean Encoding (medical_history, bmi_category, marital_status,
       smoking_status)
    7. Engineered risk features (risk_smoke, health_risk_score,
       weighted_risk_score)
    8. StandardScaler on the 11 `scaling_cols`
    9. Column pruning (income_lakhs, medical_history, bmi_category,
       risk_smoke, income_log dropped -- kept only for intermediate maths)
    10. Reindexing to the exact `expected_features` column order the model
        was fit on.

It expects the following artifacts (all produced at the end of the notebook
via `joblib.dump(...)`) to sit inside an `artifacts/` folder next to this
file:

    artifacts/
        insurance_model_v1.joblib      -> the trained StackingRegressor
        expected_features.joblib       -> list[str], final column order
        preprocessing_schema.joblib    -> {'ordinal': {...}, 'target_mean': {...}}
        scaler.joblib                  -> fitted StandardScaler
        scaling_cols.joblib            -> list[str], columns the scaler expects

If you haven't already, add this cell to the END of the notebook (after the
scaler is saved) to also persist the overall training-target mean -- it is
used as a safe fallback whenever an unseen category shows up at inference
time (this mirrors `global_mean_premium = y_train.mean()` from the notebook):

    joblib.dump(float(y_train.mean()), "artifacts/global_mean_premium.joblib")

If that file is missing, the pipeline falls back to the mean of each
target-encoding dictionary's own values, which is a very close approximation
for this dataset.
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants that mirror the notebook exactly
# --------------------------------------------------------------------------- #

RAW_COLUMNS_MAP = {
    # tolerate either the raw excel-style headers or already-clean headers
    "Age": "age",
    "Gender": "gender",
    "Region": "region",
    "Marital_status": "marital_status",
    "Number Of Dependants": "number_of_dependants",
    "BMI_Category": "bmi_category",
    "Smoking_Status": "smoking_status",
    "Employment_Status": "employment_status",
    "Income_Level": "income_level",
    "Income_Lakhs": "income_lakhs",
    "Medical History": "medical_history",
    "Insurance_Plan": "insurance_plan",
}

SMOKING_STATUS_FIX = {
    "Smoking=0": "No Smoking",
    "Does Not Smoke": "No Smoking",
    "Not Smoking": "No Smoking",
}

ONE_HOT_COLUMNS = ["gender", "region"]

# Every dummy column that *could* exist after pd.get_dummies(drop_first=True)
# on the training data. Any that don't appear for a given input get created
# and filled with 0 so the matrix shape never changes.
ALL_DUMMY_COLUMNS = [
    "gender_Male",
    "region_Northwest",
    "region_Southeast",
    "region_Southwest",
]

COLUMNS_TO_DROP_AFTER_ENGINEERING = [
    "income_lakhs",
    "medical_history",
    "bmi_category",
    "risk_smoke",
    "income_log",
    "income_level",
]

# Valid domains observed in training data -- used for light validation/clipping,
# NOT for silently changing a user's answer.
VALID_AGE_RANGE = (18, 100)
VALID_INCOME_RANGE_LAKHS = (1, 100)


class InsurancePremiumPipeline:
    """Loads the saved artifacts once, then transforms raw records into the
    exact feature matrix the stacked model expects, and predicts on them."""

    def __init__(self, artifacts_dir: str = "artifacts", prefix: str = ""):
        self.artifacts_dir = artifacts_dir
        self.prefix = prefix

        # The global model has a slightly different naming convention
        model_name = "insurance_model_v1.joblib" if prefix == "" else f"{prefix}model_v1.joblib"
        self.model = self._load(model_name)
        
        self.expected_features: list[str] = self._load(f"{prefix}expected_features.joblib")
        schema: dict = self._load(f"{prefix}preprocessing_schema.joblib")
        self.ordinal_mappings: dict = schema["ordinal"]
        self.target_mean_mappings: dict = schema["target_mean"]
        
        # Load scaler (fallback to global scaler if a specific one is missing, though we expect senior_scaler to exist now)
        try:
            self.scaler = self._load(f"{prefix}scaler.joblib")
        except FileNotFoundError:
            if prefix != "":
                # Fallback to global scaler if a segmented scaler is missing
                self.scaler = self._load("scaler.joblib", override_path=True)
            else:
                raise
                
        try:
            self.scaling_cols: list[str] = self._load(f"{prefix}scaling_cols.joblib")
        except FileNotFoundError:
            if prefix != "":
                self.scaling_cols = self._load("scaling_cols.joblib", override_path=True)
            else:
                raise

        # Fallback for any category not seen during training
        try:
            self.global_mean_premium: Optional[float] = self._load(
                f"{prefix}global_mean_premium.joblib", required=False
            )
        except FileNotFoundError:
            self.global_mean_premium = None
            
        if self.global_mean_premium is None:
            # Reasonable approximation: mean of all target-encoded means
            # pooled across every target-mean column.
            pooled = [
                v
                for mapping in self.target_mean_mappings.values()
                for v in mapping.values()
            ]
            self.global_mean_premium = float(np.mean(pooled))

    # ------------------------------------------------------------------ #
    # artifact loading
    # ------------------------------------------------------------------ #
    def _load(self, filename: str, required: bool = True, override_path: bool = False):
        if override_path:
            # Fallback path looks up two directories (e.g. from v2_segmented/rest/ up to v1_global/)
            base_dir = os.path.dirname(os.path.dirname(self.artifacts_dir))
            path = os.path.join(base_dir, "v1_global", filename)
        else:
            path = os.path.join(self.artifacts_dir, filename)
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(
                    f"Missing required artifact '{filename}' in '{self.artifacts_dir}/'. "
                    "Re-run the notebook's saving cells (joblib.dump(...)) first."
                )
            return None
        return joblib.load(path)

    # ------------------------------------------------------------------ #
    # step-by-step transform (each mirrors a notebook cell)
    # ------------------------------------------------------------------ #
    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=RAW_COLUMNS_MAP)
        df.columns = df.columns.str.replace(" ", "_").str.lower()
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["number_of_dependants"] = df["number_of_dependants"].abs()
        df["smoking_status"] = df["smoking_status"].replace(SMOKING_STATUS_FIX)
        if "income_level" in df.columns:
            df = df.drop(columns=["income_level"])
        return df

    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["income_log"] = np.log1p(df["income_lakhs"])
        df["per_capita_income"] = df["income_lakhs"] / (df["number_of_dependants"] + 1)
        df["is_over_25"] = (df["age"] > 25).astype(int)
        df["is_over_45"] = (df["age"] > 45).astype(int)
        df["is_over_60"] = (df["age"] > 60).astype(int)
        
        # Inject default Genetical Risk if missing (V2 Youth model expects this)
        # 3 is chosen as it's slightly above the dataset mean (2.5), 
        # providing a safe baseline for applicants whose genetics are unknown
        if "genetical_risk" not in df.columns:
            df["genetical_risk"] = 3
            
        return df

    def _one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # pd.get_dummies with drop_first=True on a single row drops the *only* category.
        # This causes all one-hot features to become 0 for single-record inference.
        # We manually encode them here to match the training data's get_dummies schema.
        
        if "gender" in df.columns:
            df["gender_Male"] = (df["gender"] == "Male").astype(int)
            df = df.drop(columns=["gender"])
            
        if "region" in df.columns:
            df["region_Northwest"] = (df["region"] == "Northwest").astype(int)
            df["region_Southeast"] = (df["region"] == "Southeast").astype(int)
            df["region_Southwest"] = (df["region"] == "Southwest").astype(int)
            df = df.drop(columns=["region"])
            
        # Ensure fallback zeroes if somehow a column was still missed
        for col in ALL_DUMMY_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        return df

    def _ordinal_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, mapping in self.ordinal_mappings.items():
            df[col] = df[col].map(mapping)
        return df

    def _target_mean_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, mapping in self.target_mean_mappings.items():
            df[col] = df[col].map(mapping)
            df[col] = df[col].fillna(self.global_mean_premium)
        return df

    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["risk_smoke"] = df["age"] * df["smoking_status"]

        raw_health = df["bmi_category"] * df["medical_history"] * df["smoking_status"]
        df["health_risk_score"] = np.log1p(raw_health.clip(lower=0))

        raw_weighted = df["age"] * df["health_risk_score"]
        df["weighted_risk_score"] = np.log1p(raw_weighted.clip(lower=-0.999999))
        return df

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.scaling_cols] = self.scaler.transform(df[self.scaling_cols])
        return df

    def _finalise(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [c for c in COLUMNS_TO_DROP_AFTER_ENGINEERING if c in df.columns]
        df = df.drop(columns=drop_cols)
        # Reindex guarantees exact column set/order the model was trained on.
        df = df.reindex(columns=self.expected_features, fill_value=0)
        return df

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Takes a raw dataframe (one row per person, columns matching the
        original dataset schema) and returns the exact feature matrix the
        model expects."""
        df = self._standardise_columns(raw_df)
        df = self._clean(df)
        df = self._create_base_features(df)
        df = self._one_hot_encode(df)
        df = self._ordinal_encode(df)
        df = self._target_mean_encode(df)
        df = self._create_risk_features(df)
        df = self._scale(df)
        df = self._finalise(df)
        return df

    def predict(self, raw_df: pd.DataFrame) -> np.ndarray:
        X = self.transform(raw_df)
        preds = self.model.predict(X)
        # Clip to a sensible minimum to prevent regression artifacts (negative premiums)
        # The absolute minimum premium in the entire dataset is ~3501 (Bronze, healthy youth)
        return np.clip(preds, a_min=3500, a_max=None)

    def predict_single(self, record: dict) -> float:
        """Convenience wrapper for a single person, passed as a dict with
        the raw (human-readable) field names, e.g.:

            {
                "age": 29,
                "gender": "Female",
                "region": "Southeast",
                "marital_status": "Married",
                "number_of_dependants": 2,
                "bmi_category": "Normal",
                "smoking_status": "No Smoking",
                "employment_status": "Salaried",
                "income_lakhs": 12,
                "medical_history": "No Disease",
                "insurance_plan": "Silver",
            }
        """
        raw_df = pd.DataFrame([record])
        return float(self.predict(raw_df)[0])


if __name__ == "__main__":
    # quick smoke test
    pipeline = InsurancePremiumPipeline(artifacts_dir="artifacts")
    sample = {
        "age": 29,
        "gender": "Female",
        "region": "Southeast",
        "marital_status": "Married",
        "number_of_dependants": 2,
        "bmi_category": "Normal",
        "smoking_status": "No Smoking",
        "employment_status": "Salaried",
        "income_lakhs": 12,
        "medical_history": "No Disease",
        "insurance_plan": "Silver",
    }
    print("Predicted annual premium:", pipeline.predict_single(sample))
