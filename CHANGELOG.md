# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - Segmented Model Architecture
### Added
- **Model Router:** Introduced `src.inference.router.ModelRouter` to dynamically route incoming prediction traffic based on the applicant's age.
- **Youth Model:** Trained and deployed a specialized predictive model (`youth_model_v1`) to handle applicants aged 25 and under.
- **Senior Model:** Trained and deployed a specialized predictive model (`senior_model_v1`) to handle applicants over 25.
- **Enterprise Structure:** Refactored the entire repository into a standardized directory structure, cleanly separating `data/`, `notebooks/`, `artifacts/`, `src/`, and `app/`.

### Changed
- The monolithic inference pipeline was refactored to support initializing specific pre-trained model artifacts based on a prefix (e.g. `youth_` or `senior_`).
- Streamlit `app.py` was updated to invoke the `ModelRouter` instead of directly invoking the pipeline.

### Fixed
- Fixed an issue where the global V1 model was heavily biased and inaccurate when predicting premiums for the youth demographic due to lack of distinct feature representation.

## [1.0.0] - Initial Release
### Added
- Monolithic Machine Learning pipeline leveraging a StackingRegressor (XGBoost + LightGBM + RandomForest -> Ridge).
- Beautiful, highly responsive 3D-flip web interface built with pure CSS, HTML, and Streamlit components.
