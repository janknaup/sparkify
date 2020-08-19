from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.pipeline import Pipeline
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
            * Submit Upgrade
            * Thumbs Down
            * Thumbs Up
            * Upgrade
        * frequencies of log event types defined as count / period for the events above
        * 1-hot encoded list of operating systems seen on userId
        * 1-hot encoded list of browser engines seen on userId

    Events interpreted as churn are deleted from the dataset:
    * Cancellation Confirmation
    * Submit Downgrade

    Churn-related events that are deleted to avoid data leakage:
    * Cancel
    * Downgrade
    """

    def _transform(self, dataset):
        # splitting up operations into a few temporary data tables for readability
        # session statistics
        df_session_stats = dataset.select('userId', 'sessionId', 'itemInSession').groupBy('userId', 'sessionId') \
            .count().fillna(0, 'count').withColumnRenamed("count", "_count") \
            .groupBy('userId').agg(F.mean("_count").alias("avg_session_events"),
                                   F.min("_count").alias("min_session_events"),
                                   F.max("_count").alias("max_session_events"),
                                   F.sum('_count').alias("total_session_events"))
        # user interaction period stats and per user stats
        df_by_user = dataset.select('userId', 'ts', 'gender', 'level', 'registration').groupBy('userId')\
            .agg(F.min('ts').alias("ts_min"),
                 F.max('ts').alias("ts_max"),
                 F.first('gender').alias('gender'),
                 F.first('level').alias('level'),
                 F.first('registration').alias('registration'),
                 ((F.max('ts')-F.min('ts')) / (3600 * 24 * 1000)).alias('period'))
        # user browser and operating system counts
        df_os_browser = dataset.select('userId', 'userAgent',
                                       F.regexp_extract(F.col('userAgent'), ".*?(\(.*\)).*", 1).alias('os'),
                                       F.regexp_extract(F.col('userAgent'), ".*\s(.*)", 1).alias('browser'))
        df_os_onehot = df_os_browser.groupBy('userId').pivot('os')\
            .agg(F.countDistinct('userId').alias('os')).fillna(0)
        df_browser_onehot = df_os_browser.groupBy('userId').pivot('browser') \
            .agg(F.countDistinct('userId').alias('browser')).fillna(0)
        # user page counts and frequencies
        df_page_counts = dataset.select('userId', 'page').join(df_by_user.select('userId', 'period'), on='userId')\
            .groupBy('userId').pivot('page').agg(F.count('userId').alias('count'),
                                                 (F.count('userId') / F.first('period')).alias('freq')).fillna(0)\
            .drop('Cancel_count', 'Cancellation Confirmation_count', 'Downgrade_count', 'Submit Downgrade_count',
                  'Cancel_freq', 'Cancellation Confirmation_freq', 'Downgrade_freq', 'Submit Downgrade_freq')

        return df_session_stats.join(df_by_user, on='userId').join(df_page_counts, on='userId')\
            .join(df_os_onehot, on='userId').join(df_browser_onehot, on='userId')
