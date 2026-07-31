"""
Pulse Health Insurance — Prediction Pipeline
=============================================
Replicates EVERY preprocessing & feature-engineering step from the
training notebook so raw user input becomes the exact 16-feature
vector the StackingRegressor expects.

Steps (notebook order):
  1.  Build single-row DataFrame
  2.  income_log = log1p(income_lakhs)
  3.  per_capita_income = income_lakhs / (dependants + 1)
  4.  Age bracket flags: is_over_25 / 45 / 60
  5.  One-hot encode gender & region  (drop_first => Female & Northeast as base)
  6.  Ordinal map: insurance_plan {Bronze:1,Silver:2,Gold:3}
                   employment_status {Freelancer:1,Salaried:2,Self-Employed:3}
  7.  Target-mean encode: medical_history, bmi_category,
                          marital_status, smoking_status
  8.  Feature creation: risk_smoke, health_risk_score, weighted_risk_score
  9.  StandardScaler on 11 columns
  10. Drop: income_lakhs, medical_history, bmi_category, risk_smoke, income_log
  11. Reorder to expected_features
"""

import os
import sys
import traceback

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------
class CustomException(Exception):
    """Pipeline-level exception with optional traceback detail."""

    def __init__(self, error_message, error_detail=None):
        super().__init__(str(error_message))
        if error_detail is not None:
            try:
                exc_type, exc_value, exc_tb = sys.exc_info()
                if exc_tb is not None:
                    fname = exc_tb.tb_frame.f_code.co_filename
                    lineno = exc_tb.tb_lineno
                    self.error_message = (
                        f"Error in [{fname}] line [{lineno}]: {error_message}"
                    )
                else:
                    self.error_message = str(error_message)
            except Exception:
                self.error_message = str(error_message)
        else:
            self.error_message = str(error_message)

    def __str__(self):
        return self.error_message


# ---------------------------------------------------------------------------
# Prediction Pipeline
# ---------------------------------------------------------------------------
class PredictPipeline:
    """Loads saved artifacts once, then transforms raw input for prediction."""

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        art = os.path.join(base_dir, "artifacts")

        missing = [
            f for f in [
                "insurance_model_v1.joblib",
                "expected_features.joblib",
                "preprocessing_schema.joblib",
                "scaler.joblib",
                "scaling_cols.joblib",
            ]
            if not os.path.exists(os.path.join(art, f))
        ]
        if missing:
            raise CustomException(
                f"Missing artifact files: {missing}. "
                "Run generate_artifacts.py first."
            )

        self.model            = joblib.load(os.path.join(art, "insurance_model_v1.joblib"))
        self.expected_features = joblib.load(os.path.join(art, "expected_features.joblib"))
        self.schema           = joblib.load(os.path.join(art, "preprocessing_schema.joblib"))
        self.scaler           = joblib.load(os.path.join(art, "scaler.joblib"))
        self.scaling_cols     = joblib.load(os.path.join(art, "scaling_cols.joblib"))

    # ------------------------------------------------------------------ #
    def predict(self, user_data: dict) -> float:
        """
        user_data keys (raw human-readable values):
            age, gender, region, marital_status, number_of_dependants,
            bmi_category, smoking_status, employment_status,
            income_lakhs, medical_history, insurance_plan
        Returns predicted annual premium (float, in INR).
        """
        try:
            df = self._build_dataframe(user_data)
            df = self._create_income_features(df)
            df = self._create_age_flags(df)
            df = self._one_hot_encode(df)
            df = self._ordinal_encode(df)
            df = self._target_mean_encode(df)
            df = self._feature_creation(df)
            df = self._scale(df)
            df = self._drop_and_reorder(df)
            return float(self.model.predict(df)[0])
        except CustomException:
            raise
        except Exception as exc:
            raise CustomException(str(exc)) from exc

    # ------------------------------------------------------------------ #
    # Private helpers — one per notebook step
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_dataframe(user_data: dict) -> pd.DataFrame:
        """Step 1 — single-row DataFrame with lowercase notebook column names."""
        return pd.DataFrame({
            "age":                  [int(user_data["age"])],
            "gender":               [user_data["gender"]],
            "region":               [user_data["region"]],
            "marital_status":       [user_data["marital_status"]],
            "number_of_dependants": [int(user_data["number_of_dependants"])],
            "bmi_category":         [user_data["bmi_category"]],
            "smoking_status":       [user_data["smoking_status"]],
            "employment_status":    [user_data["employment_status"]],
            "income_lakhs":         [float(user_data["income_lakhs"])],
            "medical_history":      [user_data["medical_history"]],
            "insurance_plan":       [user_data["insurance_plan"]],
        })

    @staticmethod
    def _create_income_features(df: pd.DataFrame) -> pd.DataFrame:
        """Steps 2-3 — income_log & per_capita_income."""
        df = df.copy()
        df["income_log"]       = np.log1p(df["income_lakhs"])
        df["per_capita_income"] = df["income_lakhs"] / (df["number_of_dependants"] + 1)
        return df

    @staticmethod
    def _create_age_flags(df: pd.DataFrame) -> pd.DataFrame:
        """Step 4 — cumulative age bracket flags."""
        df = df.copy()
        df["is_over_25"] = (df["age"] > 25).astype(int)
        df["is_over_45"] = (df["age"] > 45).astype(int)
        df["is_over_60"] = (df["age"] > 60).astype(int)
        return df

    @staticmethod
    def _one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 5 — manual one-hot encoding matching pd.get_dummies(drop_first=True).
        Gender  baseline = Female  => gender_Male
        Region  baseline = Northeast => region_Northwest/Southeast/Southwest
        """
        df = df.copy()
        gender = df["gender"].iloc[0]
        region = df["region"].iloc[0]

        df["gender_Male"]       = int(gender == "Male")
        df["region_Northwest"]  = int(region == "Northwest")
        df["region_Southeast"]  = int(region == "Southeast")
        df["region_Southwest"]  = int(region == "Southwest")

        df = df.drop(columns=["gender", "region"])
        return df

    def _ordinal_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 6 — ordinal mapping from schema."""
        df = df.copy()
        for col, mapping in self.schema["ordinal"].items():
            df[col] = df[col].map(mapping)
        return df

    def _target_mean_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 7 — replace categories with their training-set group mean premium."""
        df = df.copy()
        for col, mapping in self.schema["target_mean"].items():
            df[col] = df[col].map(mapping)
            if df[col].isna().any():
                # Fallback: use average of all group means
                fallback = float(np.mean(list(mapping.values())))
                df[col] = df[col].fillna(fallback)
        return df

    @staticmethod
    def _feature_creation(df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 8 — engineered interaction features.
        At this point medical_history, bmi_category, marital_status,
        and smoking_status are already numeric (target-mean encoded).
        """
        df = df.copy()
        df["risk_smoke"] = df["age"] * df["smoking_status"]

        raw_health = df["bmi_category"] * df["medical_history"] * df["smoking_status"]
        df["health_risk_score"] = np.log1p(raw_health)

        raw_weighted = df["age"] * df["health_risk_score"]
        df["weighted_risk_score"] = np.log1p(raw_weighted)
        return df

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 9 — apply StandardScaler to the same 11 columns as training."""
        df = df.copy()
        df[self.scaling_cols] = self.scaler.transform(df[self.scaling_cols])
        return df

    def _drop_and_reorder(self, df: pd.DataFrame) -> pd.DataFrame:
        """Steps 10-11 — drop VIF-pruned columns, reorder to expected_features."""
        df = df.copy()
        to_drop = ["income_lakhs", "medical_history", "bmi_category",
                   "risk_smoke", "income_log"]
        df = df.drop(columns=to_drop)
        return df[self.expected_features]
