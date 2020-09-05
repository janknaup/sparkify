import json
import plotly

from flask import Flask
from flask import render_template, request, jsonify

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import LinearSVC
from model import LogCleanTransformer, UserLogTransformer, TrainingAssembler
import plotly.express as px
import plotly.graph_objects as go

spark = SparkSession \
    .builder \
    .appName("Churn") \
    .getOrCreate()



app = Flask(__name__)
model = LinearSVC.load('lsvc_model')



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    categories = list(df.columns)
    for remcol in ["message", "original", "id", "genre"]:
        categories.remove(remcol)

    category_counts = df[categories].sum().values
    english_category_counts = df.loc[df['original'].isna().values][categories].sum().values

    genre_category_counts = df.groupby('genre').sum()[categories]

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories,
                    y=category_counts - english_category_counts,
                    name='Other'
                ),
                Bar(
                    x=categories,
                    y=english_category_counts,
                    name='English'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories<br>By Original Message Language',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                },
                'barmode': 'stack'
            }
        },
        {
            'data': [
                Bar(
                    x=categories,
                    y=100 * genre_category_counts.loc[genre] / category_counts,
                    name=genre
                ) for genre in genre_names
            ],

            'layout': {
                'title': 'Proportion of Message Genres by Category',
                'yaxis': {
                    'title': "Genre Percentage"
                },
                'xaxis': {
                    'title': "Category"
                },
                'barmode': 'stack'
            }
        }
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
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

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