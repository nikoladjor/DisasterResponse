import json
import plotly
import plotly.express as px
import pandas as pd



from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

import joblib
from sqlalchemy import create_engine
import sys

sys.path.append('../')
from models.train_classifier import tokenize, acc_score

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse', engine)

# Report
report = joblib.load('../models/production_model_report.joblib')
labels = ['category','precision', 'recall', 'f1_score', 'support']
report_df = pd.DataFrame.from_records(report[:-1], columns=labels)


# load model
# MODEL_PATH = "../models/dev_classifier_low_estimators.joblib"
MODEL_PATH = "../models/classifier_production.joblib"

model = joblib.load(MODEL_PATH)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_counts = genre_counts.reset_index()

    fig1 = px.bar(data_frame=genre_counts, x='genre', y='message')

    fig_precision = px.bar(data_frame=report_df, x='category', y='precision', color='precision', 
                  labels={'category': "Category", 'precision': "Precision"} )
    
    fig_recall = px.bar(data_frame=report_df, x='category', y='recall', color='recall', 
                  labels={'category': "Category", 'recall': "Recall"} )
    
    fig_f1 = px.bar(data_frame=report_df, x='category', y='f1_score', color='f1_score', 
                  labels={'category': "Category", 'f1_score': "F1 Score"} )

    
    graphs = [fig1, fig_precision, fig_recall, fig_f1]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, accuracy=report[-1][1])

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()