# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:00:52 2022

@author: pierr
"""
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import json
import joblib
import plotly.graph_objects as go
import pandas as pd
import requests
import plotly.express as px
import numpy as np


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
feats = requests.get("https://creditmanager2.herokuapp.com/feats/").json()

app = Dash(__name__, external_stylesheets=external_stylesheets)
server= app.server

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                feats,
                'EXT_SOURCE_3',
                id='crossfilter-feature',
            )
        ],
        style={'width': '66%', 'display': 'inline-block'}),
        
        
        
        
        
        html.Div([
            dcc.Input(id="client_id", type="number", value='100001'),
            html.Button(id="validation_bt", n_clicks=0, children="Valider")
        ],
        style={'width': '33%', "float" : "right", 'display': 'inline-block', 'background-color' : 'black'})
    ]),
    html.Div(dcc.Graph(id='bar_mean', figure = {"layout" : {"height" : 800}}), 
        style={'width': '33%', 'display': 'inline-block', 'padding': '0 20', "vertical_align" : "middle", 'background-color' : 'black'}),
    html.Div(dcc.Graph(id='boxplot', figure = {"layout" : {"height" : 800}}), 
        style={'width': '33%', 'display': 'inline-block', 'padding': '0 20', "vertical_align" : "middle", 'background-color' : 'black'}),
    html.Div([
        dcc.Graph(id='score'),
        dcc.Graph(id='feature_imp')
        ], style={'display': 'inline-block', 'width': '33%', "float":"right", 'background-color' : 'black'})
])
        

@app.callback(Output('score', 'figure'),
              Input('validation_bt', 'n_clicks'),
              State('client_id', 'value'))
def update_score(n_clicks, client_id):

    score_min = requests.get("https://creditmanager2.herokuapp.com/score_min/").json()["score_min"] * 100
        
    r = requests.get("https://creditmanager2.herokuapp.com/predict", params={"client_id" : client_id})
    val = r.json()["proba"] * 100
    
    if val > score_min:
        accept = "Accept??"
        color = "darkgreen"
    else:
        accept = "Refus??"
        color = "darkred"
    fig1 = go.Figure()
    
    fig1.add_trace(go.Indicator(
        domain = {"x" : [0,1], "y" : [0,1]},
        
        title = {"text" : "Score", "font_size" : 40},
        value = val,
        number = {"font_size" : 50},
        
        mode = "gauge + number",
        
        gauge = {
            "shape" : "angular",
            "steps" : [
                {"range" : [0, score_min], "color" : "red"},
                {"range" : [score_min, 100], "color" : "green"}
                ],
            "bar" : {"color" : "black", "thickness" : 0.5},
            "axis" : {"range" : [None, 100]}
            }
        )
    )
    fig1.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background']
    )
    fig1.add_annotation(x=0.5, y=0.4, text=accept, font = dict(size = 30, color=color), showarrow = False)
    
    return fig1
      

@app.callback(Output('feature_imp', 'figure'),
              Input('validation_bt', 'n_clicks'),
              State('client_id', 'value'))
def update_fi(n_clicks, client_id):
    shap_vals = requests.get("https://creditmanager2.herokuapp.com/importances", params={"client_id" : client_id}).json()
    
    df_feats = pd.DataFrame(shap_vals, columns=["importances"])
    df_feats["feats"] = feats
    df_feats["abs"] = abs(df_feats["importances"])
    df_feats["Influence"]= np.where(df_feats["importances"]<0, "Negative", "Positive")
    df_feats.sort_values(by="abs", ascending=False, inplace=True)
    df_feats.drop(columns=["abs"], inplace=True)
    
    fig2 = px.bar(df_feats.iloc[:10],
           x= "importances",
           y = "feats", 
           color = "Influence",
           orientation="h",
           title = "Principales donn??es influant sur le r??sultat")
    fig2.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background']
    )
    fig2.update_xaxes(title="Impact sur le r??sultat")
    fig2.update_yaxes(title="Variable ??tudi??e")
    
    return fig2

@app.callback(Output('crossfilter-feature', 'value'),
              Input('feature_imp', 'clickData'))
def change_feat(clickdata):
    if clickdata == None:
        return "EXT_SOURCE_3"
    else:
        return clickdata["points"][0]["y"]
        
@app.callback(Output('bar_mean', 'figure'),
              Input('validation_bt', 'n_clicks'),
              Input("crossfilter-feature", "value"),
              State('client_id', 'value'))
def plot_bar(n_clicks, feature, client_id):
    results = requests.get("https://creditmanager2.herokuapp.com/bar", 
                            params={"client_id" : client_id, "feature" : feature}).json()                      
        
    fig3 = px.bar(
           x = ["client", "moyenne"],
           y = [results[0], results[1]],
           color = [results[0], results[1]],
           title = "Comparaison du client ?? la moyenne")
    fig3.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background']
    )
    fig3.update_xaxes(title="")
    fig3.update_yaxes(title="Valeur")
    
    return fig3
    
@app.callback(Output('boxplot', 'figure'),
              Input("crossfilter-feature", "value"))
def plot_box(feature):
    dff = requests.get("https://creditmanager2.herokuapp.com/boxplot", 
                            params={"feature" : feature}).json()  
    
    fig4 = px.box(dff, title = "R??partition de la variable dans la client??le")
    fig4.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background']
    )
    fig4.update_xaxes(title="")
    fig4.update_yaxes(title="Valeur")
        
    return fig4

