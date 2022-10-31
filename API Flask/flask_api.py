import joblib
from flask import Flask, request
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

model = joblib.load('pipeline_lreg.joblib')

explainer = joblib.load('explainer.joblib')

@app.route("/prediction", methods=['POST'])
def post_prediction():
    user_id_data = json.loads(request.data)
    values=user_id_data['data']
  
    prediction = model.predict_proba(values).tolist()
    
    return {
        "predict_proba" : prediction
    }

@app.route("/shap_values", methods=['POST'])
def post_shap_values():
    user_id_data = json.loads(request.data)
    
    df_user_id = pd.DataFrame(data=user_id_data["data"], index=user_id_data["index"], columns=user_id_data["columns"])
    
    df_user_id = df_user_id.fillna(np.nan)
    
    shap_values = explainer.shap_values(df_user_id)
    
    df_feature_importance_class_0 = pd.DataFrame(data=shap_values[0], index=user_id_data["index"], columns=user_id_data["columns"])
    
    result = df_feature_importance_class_0.to_json(orient="split")
    
    return result
            