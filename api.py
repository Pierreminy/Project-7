# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:27:38 2022

@author: pierr
"""
from flask import Flask, request
import pickle 
import json
import lightgbm
import gzip
import joblib
import numpy as np
app = Flask(__name__)



feats = np.genfromtxt('feats.csv', dtype='unicode', delimiter=',')


def load_model(file_path):
 
    with open(file_path, 'rb') as f:
        classifier = pickle.load(f)
    return classifier



def load_joblib(file_path):
 
    with open(file_path, 'rb') as f:
        classifier = joblib.load(f)
    return classifier



def load_data(file_path):
 
    with gzip.open(file_path, 'rb') as f:
        classifier = pickle.load(f)
    return classifier

model = load_model(r'D:\API\model.pkl')
test_df = load_data(r'D:\API\test_df.gz')
explainer = load_joblib(r'D:\API\shap_explainer.joblib')


test_df.drop(columns=["index"], inplace=True)
test_df.set_index("SK_ID_CURR", inplace=True)

def make_prediction(client_id):
    return model.predict_proba([test_df[feats].loc[client_id]])[0, 1]
    
def explain(client_id):
    return explainer.shap_values(test_df[feats].loc[client_id].to_numpy().reshape(1, -1))[1][0][:]


@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/', methods=['GET'])
def index():
    return {'message': 'Hello, stranger'}


@app.route('/score_min/', methods=['GET'])
def score_min():
    return {"score_min" : 0.55} 


@app.route('/predict', methods=["GET"])
def proba():
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        pred = make_prediction(client_id)
        return {"proba" : pred}
    else:
        return "Error"

@app.route('/feats/', methods=["GET"])
def feats_ret():
    return json.dumps(feats.tolist()) 

@app.route('/importances', methods=["GET"])
def importances():
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        shap_vals = explain(client_id).tolist()
        return json.dumps(shap_vals)
    else:
        return "Error"    

@app.route('/bar', methods=["GET"])
def bar():
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        feat = str(request.args["feature"])
        
        dff = test_df[feat]
        retour = []
        retour.append(float(dff.loc[client_id]))
        retour.append(np.mean(dff))
        del dff      
     
        return json.dumps(retour)
    else:
        return "Error"    
    
@app.route('/boxplot', methods=["GET"])
def boxplot():
    if 'feature' in request.args:
        feat = str(request.args["feature"])
        
        dff = test_df[feat]
     
        return json.dumps(dff.tolist())
    else:
        return "Error"   
    
    

app.run
