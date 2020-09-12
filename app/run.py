import json
import plotly

from flask import Flask
from flask import render_template, request, jsonify

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from model import FeatureUnassembler, FEATURE_COLUMNS, KNOWN_BROWSERS, KNOWN_MESSAGES, KNOWN_OS
from model import LogCleanTransformer, UserLogTransformer, TrainingAssembler
from pyspark.ml.classification import LinearSVCModel
from pyspark.sql.types import StructType, StructField, LongType, IntegerType
from pyspark.ml.linalg import VectorUDT
from pyspark import SparkFiles

import plotly.graph_objects as gob

spark = SparkSession \
    .builder \
    .appName("Churn_Dashboard") \
    .getOrCreate()



app = Flask(__name__)
spark.sparkContext.addFile('lsvc_model', recursive=True)
spark.sparkContext.addFile('mini_sparkify_event.json', recursive=True)
model = LinearSVCModel.load(SparkFiles.get('lsvc_model'))
# load training data set stats
df_pred = spark.read.json('results_model/lsvc-prediction.json')
schema = StructType([StructField('userId', LongType(), False),
                     StructField('features', VectorUDT(), False),
                     StructField('label', IntegerType(), True)])
df_train = spark.read.json('results_model/traindata.json', schema=schema)
df_test = spark.read.json('results_model/testdata.json', schema=schema)
df_all = df_train.union(df_test)
# restore feature columns from features vector
feature_una = FeatureUnassembler()
unassembled = feature_una.transform(df_all).join(df_all.select('userId', 'label'), on='userId')
# count stats
unassembled_bylabel = unassembled.groupBy('label')
hist_dist = unassembled_bylabel.agg(*[(F.sum(col)/F.count(col)).alias(col) for col in KNOWN_OS + KNOWN_BROWSERS])
hist_dist.cache()
# prepare dashboard plot data
# histogram raw data
kept_stats = unassembled.select('label', 'Roll Advert_freq', 'period', 'NextSong_freq',
                                'maxLevel', 'changedLevel').filter(F.col('label') == 0).collect()
churn_stats = unassembled.select('label', 'Roll Advert_freq', 'period', 'NextSong_freq',
                                 'maxLevel', 'changedLevel').filter(F.col('label') == 1).collect()
# operating system distribution
od_kept = hist_dist.filter(F.col('label') == 0).collect()[0].asDict()
kept_os = [od_kept[browser] for browser in KNOWN_OS]
od_churn = hist_dist.filter(F.col('label') == 1).collect()[0].asDict()
churn_os = [od_churn[browser] for browser in KNOWN_OS]
# browser distribution
bd_kept = hist_dist.filter(F.col('label') == 0).collect()[0].asDict()
kept_browsers = [bd_kept[browser] for browser in KNOWN_BROWSERS]
bd_churn = hist_dist.filter(F.col('label') == 1).collect()[0].asDict()
churn_browsers = [bd_churn[browser] for browser in KNOWN_BROWSERS]



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

        {
            'data': [
                gob.Bar(name="kept", x=KNOWN_OS, y=kept_os),
                gob.Bar(name="churned", x=KNOWN_OS, y=churn_os)
            ],

            'layout': {
                'title': 'Operating System Distribution',
                'yaxis': {
                    'title': "User Fraction"
                },
                'xaxis': {
                    'title': "Operating System"
                },
                'barmode': 'group',
            }
        },

        {
            'data': [
                gob.Bar(name="kept", x=KNOWN_BROWSERS, y=kept_browsers),
                gob.Bar(name="churned", x=KNOWN_BROWSERS, y=churn_browsers)
            ],

            'layout': {
                'title': 'Browser Distribution',
                'yaxis': {
                    'title': "User Fraction"
                },
                'xaxis': {
                    'title': "Browser"
                },
                'barmode': 'group',
            }
        },

        {
            'data': [
                gob.Histogram(x=[kaf['NextSong_freq'] for kaf in kept_stats], histnorm='percent',
                              name='kept'),
                gob.Histogram(x=[caf['NextSong_freq'] for caf in churn_stats], histnorm='percent',
                              name='churned')
            ],

            'layout': {
                'title': 'Distribution of "Next Song" Frequencies per User',
                'yaxis': {
                    'title': "User Fraction (%)"
                },
                'xaxis': {
                    'title': "Next Song Events Per Month"
                },
                'barmode': 'group',
            }
        },

        {
            'data': [
                gob.Histogram(x=[kaf['Roll Advert_freq'] for kaf in kept_stats], histnorm='percent',
                              name='kept'),
                gob.Histogram(x=[caf['Roll Advert_freq'] for caf in churn_stats], histnorm='percent',
                              name='churned')
            ],

            'layout': {
                'title': 'Distribution of Advert Frequencies per User',
                'yaxis': {
                    'title': "User Fraction (%)", 'type': 'log'
                },
                'xaxis': {
                    'title': "Adverts Per Month"
                },
                'barmode': 'group',
            }
        },

        {
            'data': [
                gob.Histogram(x=[kaf['period'] for kaf in kept_stats], histnorm='percent',
                              name='kept'),
                gob.Histogram(x=[caf['period'] for caf in churn_stats], histnorm='percent',
                              name='churned')
            ],

            'layout': {
                'title': 'Distribution of Periods per User',
                'yaxis': {
                    'title': "User Fraction (%)"
                },
                'xaxis': {
                    'title': "Usage Period (Months)"
                },
                'barmode': 'group',
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
    data_schema = "artist STRING, auth STRING, firstName STRING, gender STRING, itemInSession INT, lastName STRING," \
                  "length DOUBLE, level STRING, location STRING, method STRING, page STRING, registration LONG," \
                  "sessionId INT, song STRING, status int, ts LONG, userAgent STRING, userId STRING"
    df_data = spark.read.json(SparkFiles.get('mini_sparkify_event.json'), schema=data_schema)\
        .withColumn('iuid', F.col('userId').cast('long')).drop('userId').withColumnRenamed('iuid', 'userId')
    cleaner = LogCleanTransformer()
    mtr = UserLogTransformer()
    ta = TrainingAssembler()
    df_data_pred = model.transform(ta.transform(mtr.transform(cleaner.transform(df_data))))
    classification_labels = df_data_pred.select('userId', 'prediction').orderBy('userId').collect()
    classification_results = [(lab['userId'], lab['prediction']) for lab in classification_labels]

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