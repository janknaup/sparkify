import json
import plotly

from flask import Flask
from flask import render_template, request, jsonify

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import LinearSVC
from model import LogCleanTransformer, UserLogTransformer, TrainingAssembler
import plotly.express as px
import plotly.graph_objects as gob

spark = SparkSession \
    .builder \
    .appName("Churn_Dashboard") \
    .getOrCreate()



app = Flask(__name__)
#model = LinearSVC.load('lsvc_model')
df_pred = spark.read.json('results_model/lsvc-prediction.json')
df_train = spark.read.json('results_model/traindata.json')
df_test = spark.read.json('results_model/testdata.json')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    confusion_counts = df_pred.withColumn('true negative', ((1.0 - F.col('prediction')) * (1.0 - F.col('label'))))\
        .agg(F.sum('tp').alias('true positive'), F.sum('fp').alias('false positive'),
                F.sum('fn').alias('false negative'), F.sum('true negative').alias('true negative'))

    confusion_names = confusion_counts.columns
    confusion_values = [confusion_counts.collect()[0][col] for col in confusion_names]

    # create visuals
    graphs = [
        {
            'data': [
                gob.Pie(
                    labels=confusion_names,
                    values=confusion_values
                )
            ],

            'layout': {
                'title': 'Test Data Prediction Quality',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Classification Result"
                }
            }
        },

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    #classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df_pred.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()