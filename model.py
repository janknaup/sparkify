from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Transformer


class UserLogTransformer(Transformer):
    """
    Custom transformer that aggregates Spotify log data into user features for predicting
    if they are at risk of churning.

    Extracted user information are:

    * user base data:
        * gender
        * level
        * registration (time tamp)
    * aggregated log data
        * avg_session_events  mean number of log events per session
        * min_session_events
        * max_session_events
        * total_session_events
        * ts_min  timestamp of earliest log event
        * ts_max  timestamp of most recent log event
        * period  interval in days between earliest and latest log event
        * counts of log event types (pages)
            * About
            * Add Friend
            * Add to Playlist
            * Error
            * Help
            * Home
            * Logout
            * NextSong
            * Roll Advert
            * Save Settings
            * Settings
            * Thumbs Down
            * Thumbs Up
            * Upgrade
            * Submit Downgrade
            * Submit Upgrade
        * frequencies of log event types defined as count / period for the events above
        * 1-hot encoded list of operating systems seen on userId
        * 1-hot encoded list of browser engines seen on userId

    Events interpreted as churn are deleted from the dataset:
    * Cancellation Confirmation

    Churn-related events that are deleted to avoid data leakage:
    * Cancel
    """

    def _transform(self, dataset):
        # splitting up operations into a few temporary data tables for readability
        # session statistics
        df_session_stats = dataset.select('userId', 'sessionId', 'itemInSession').groupBy('userId', 'sessionId') \
            .count().fillna(0, 'count').withColumnRenamed("count", "_count") \
            .groupBy('userId').agg(F.mean("_count").cast('double').alias("avg_session_events"),
                                   F.min("_count").cast('double').alias("min_session_events"),
                                   F.max("_count").cast('double').alias("max_session_events"),
                                   F.sum('_count').cast('double').alias("total_session_events"))
        # user interaction period stats and per user stats
        df_user_categories = dataset.select('userId', F.when(dataset.gender == 'f', 1.0).otherwise(0.0).alias('_gender'),
                                            F.when(dataset.level == 'paid', 1.0).otherwise(0.0).alias('_level'))\
            .groupBy('userId').agg(F.first('_gender').alias('gender'), F.max('_level').alias('maxLevel'),
                                   (F.max('_level') - F.min('_level')).alias('changedLevel'))
        df_by_user = dataset.select('userId', 'ts', 'registration').groupBy('userId')\
            .agg(#F.min('ts').cast('double').alias("ts_min"),
                 #F.max('ts').cast('double').alias("ts_max"),
                 F.first('registration').cast('double').alias('registration'),
                 ((F.max('ts')-F.min('ts')) / (3600 * 24 * 1000)).alias('period')
                 )
        # user browser and operating system counts
        df_os_browser = dataset.select('userId', 'userAgent',
                                       F.regexp_replace(F.regexp_extract(F.col('userAgent'), ".*?(\(.*\)).*", 1),
                                        '[\(\);:;\s\/.,]+', '').alias('os'),
                                       F.regexp_replace(F.regexp_extract(F.col('userAgent'), ".*\s(.*)", 1),
                                        '[\(\);:;\s\/.,]+', '').alias('browser'))
        df_os_onehot = df_os_browser.groupBy('userId').pivot('os')\
            .agg(F.countDistinct('userId').cast('double').alias('os')).fillna(0)
        df_browser_onehot = df_os_browser.groupBy('userId').pivot('browser') \
            .agg(F.countDistinct('userId').cast('double').alias('browser')).fillna(0)
        # user page counts and frequencies
        df_page_counts = dataset.select('userId', F.column('page').alias('page'))\
            .join(df_by_user.select('userId', 'period'), on='userId')\
            .groupBy('userId').pivot('page').agg(F.count('userId').cast('double').alias('count'),
                                                 (F.count('userId') / F.first('period')).cast('double')
                                                 .alias('freq')).fillna(0)\
            .drop('Cancel_count', 'Cancellation Confirmation_count', 'Submit Downgrade_count',
                  'Cancel_freq', 'Cancellation Confirmation_freq', 'Submit Downgrade_freq')
        return df_by_user.join(df_page_counts, on='userId')\
            .join(df_user_categories, on='userId')\
            .join(df_os_onehot, on='userId')\
            .join(df_browser_onehot, on='userId')\
            .join(df_session_stats, on='userId')


class LogCleanTransformer(Transformer):
    """
    Custom Transformer that cleans user log data for machine learning.

    Drops all rows with nan userId values
    """

    def _transform(self, dataset):
        return dataset.dropna(how='any', subset=['userId', ])


class UserLabelTransformer(Transformer):
    """
    Custom Transformer that returns if a user has churn events in his log history

    Defined Churn events are:
    * Cancellation Confirmation
    * Submit Downgrade
    """

    def _transform(self, dataset):
        return dataset.select('userId', F.when((dataset.page == 'Cancellation Confirmation'),
                                               1).otherwise(0).alias('churn'))\
            .groupBy('userId').agg(F.max('churn').alias('churned'))


class TrainingAssembler(Transformer):
    """"
    Assemble the data set for training into label and feature vector columns.

    Expects a dataframe of numeric columns, one of which should be named 'churned'. The 'churned' column is used as the
    labels columns. All other columns are assembled into the features vector.
    """

    def _transform(self, dataset):
        input_cols = dataset.columns
        input_cols.remove('churned')
        vectassemble = VectorAssembler(inputCols=input_cols, outputCol='features', handleInvalid='skip')
        return vectassemble.transform(dataset).select(
            F.column('churned').alias('label'), 'features', 'userId'
        )


class MasterTransformer(Transformer):
    """
    Transformer instance that puts together the building blocks for label and feature extraction
    """

    def _transform(self, dataset):
        logtransform = UserLogTransformer()
        labeltransform = UserLabelTransformer()
        assembler = TrainingAssembler()
        return assembler.transform(logtransform.transform(dataset).join(labeltransform.transform(dataset), on='userId'))
