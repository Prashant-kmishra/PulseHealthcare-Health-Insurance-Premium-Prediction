import os
from typing import Dict, Any, Union
from src.inference.pipeline import InsurancePremiumPipeline

class ModelRouter:
    """
    Routes prediction requests to either the Youth model or the Senior/Rest model
    based on the applicant's age.
    """
    
    def __init__(self, base_artifacts_dir: str):
        """
        Initializes both pipelines and loads their respective models into memory.
        """
        youth_dir = os.path.join(base_artifacts_dir, "v2_segmented", "youth")
        senior_dir = os.path.join(base_artifacts_dir, "v2_segmented", "rest")
        
        self.youth_pipeline = InsurancePremiumPipeline(artifacts_dir=youth_dir, prefix="youth_")
        self.senior_pipeline = InsurancePremiumPipeline(artifacts_dir=senior_dir, prefix="senior_")
        
    def predict(self, user_data: Dict[str, Any]) -> Union[float, list]:
        """
        Routes the user data to the appropriate model based on age.
        Youth is defined as age <= 35 to prevent the global/senior linear
        models from catastrophically under-predicting premiums for healthy young adults.
        """
        age = user_data.get("age", 25)
        
        # Ensure age is an integer
        try:
            age = int(age)
        except (ValueError, TypeError):
            age = 25
            
        if age <= 35:
            # Route to youth model
            return self.youth_pipeline.predict_single(user_data)
        else:
            # Route to senior/rest model
            return self.senior_pipeline.predict_single(user_data)
