from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Transformer
from pyspark.sql.types import DoubleType, ArrayType

KNOWN_BROWSERS = ['Firefox240',
                  'Firefox300',
                  'Firefox310',
                  'Firefox320',
                  'Gecko',
                  'Safari5345910"',
                  'Safari53736"',
                  'Safari537749"',
                  'Safari5377514"',
                  'Safari537764"',
                  'Safari537774"',
                  'Safari537782"',
                  'Safari60013"',
                  'Safari60018"',
                  'Safari953753"',
                  'Trident50',
                  'Trident60']

KNOWN_OS = ['MacintoshIntelMacOSX106rv310',
            'MacintoshIntelMacOSX107rv310',
            'MacintoshIntelMacOSX108rv310',
            'MacintoshIntelMacOSX109rv300',
            'MacintoshIntelMacOSX109rv310',
            'MacintoshIntelMacOSX10_10AppleWebKit60013KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_10AppleWebKit60018KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_10_0AppleWebKit53736KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_6_8AppleWebKit5345910KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_6_8AppleWebKit53736KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_7_5AppleWebKit53736KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_7_5AppleWebKit537774KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_8_5AppleWebKit53736KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_8_5AppleWebKit537774KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_9_2AppleWebKit53736KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_9_2AppleWebKit537749KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_9_2AppleWebKit5377514KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_9_3AppleWebKit53736KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_9_3AppleWebKit537764KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_9_4AppleWebKit53736KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_9_4AppleWebKit537774KHTMLlikeGecko',
            'MacintoshIntelMacOSX10_9_4AppleWebKit537782KHTMLlikeGecko',
            'WindowsNT51AppleWebKit53736KHTMLlikeGecko',
            'WindowsNT51rv310',
            'WindowsNT60rv310',
            'WindowsNT61AppleWebKit53736KHTMLlikeGecko',
            'WindowsNT61WOW64AppleWebKit53736KHTMLlikeGecko',
            'WindowsNT61WOW64Trident70rv110',
            'WindowsNT61WOW64rv240',
            'WindowsNT61WOW64rv300',
            'WindowsNT61WOW64rv310',
            'WindowsNT61WOW64rv320',
            'WindowsNT61rv310',
            'WindowsNT62WOW64AppleWebKit53736KHTMLlikeGecko',
            'WindowsNT62WOW64rv310',
            'WindowsNT63WOW64AppleWebKit53736KHTMLlikeGecko',
            'WindowsNT63WOW64Trident70rv110',
            'WindowsNT63WOW64rv310',
            'X11Linuxx86_64AppleWebKit53736KHTMLlikeGecko',
            'X11Linuxx86_64rv310',
            'X11UbuntuLinuxi686rv310',
            'X11UbuntuLinuxx86_64rv300',
            'X11UbuntuLinuxx86_64rv310',
            'compatibleMSIE100WindowsNT61WOW64Trident60',
            'compatibleMSIE100WindowsNT62WOW64Trident60',
            'compatibleMSIE90WindowsNT61Trident50',
            'compatibleMSIE90WindowsNT61WOW64Trident50',
            'iPadCPUOS7_1_1likeMacOSXAppleWebKit537512KHTMLlikeGecko',
            'iPadCPUOS7_1_2likeMacOSXAppleWebKit537512KHTMLlikeGecko',
            'iPhoneCPUiPhoneOS7_1_1likeMacOSXAppleWebKit537512KHTMLlikeGecko',
            'iPhoneCPUiPhoneOS7_1_2likeMacOSXAppleWebKit537512KHTMLlikeGecko',
            'iPhoneCPUiPhoneOS7_1likeMacOSXAppleWebKit537512KHTMLlikeGecko']

FEATURE_COLUMNS = ['userId',
                   'registration',
                   'period',
                   'About_count',
                   'About_freq',
                   'Add Friend_count',
                   'Add Friend_freq',
                   'Add to Playlist_count',
                   'Add to Playlist_freq',
                   'Downgrade_count',
                   'Downgrade_freq',
                   'Error_count',
                   'Error_freq',
                   'Help_count',
                   'Help_freq',
                   'Home_count',
                   'Home_freq',
                   'Logout_count',
                   'Logout_freq',
                   'NextSong_count',
                   'NextSong_freq',
                   'Roll Advert_count',
                   'Roll Advert_freq',
                   'Save Settings_count',
                   'Save Settings_freq',
                   'Settings_count',
                   'Settings_freq',
                   'Submit Upgrade_count',
                   'Submit Upgrade_freq',
                   'Thumbs Down_count',
                   'Thumbs Down_freq',
                   'Thumbs Up_count',
                   'Thumbs Up_freq',
                   'Upgrade_count',
                   'Upgrade_freq',
                   'gender',
                   'maxLevel',
                   'changedLevel',
                   'MacintoshIntelMacOSX106rv310',
                   'MacintoshIntelMacOSX107rv310',
                   'MacintoshIntelMacOSX108rv310',
                   'MacintoshIntelMacOSX109rv300',
                   'MacintoshIntelMacOSX109rv310',
                   'MacintoshIntelMacOSX10_10AppleWebKit60013KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_10AppleWebKit60018KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_10_0AppleWebKit53736KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_6_8AppleWebKit5345910KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_6_8AppleWebKit53736KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_7_5AppleWebKit53736KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_7_5AppleWebKit537774KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_8_5AppleWebKit53736KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_8_5AppleWebKit537774KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_9_2AppleWebKit53736KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_9_2AppleWebKit537749KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_9_2AppleWebKit5377514KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_9_3AppleWebKit53736KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_9_3AppleWebKit537764KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_9_4AppleWebKit53736KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_9_4AppleWebKit537774KHTMLlikeGecko',
                   'MacintoshIntelMacOSX10_9_4AppleWebKit537782KHTMLlikeGecko',
                   'WindowsNT51AppleWebKit53736KHTMLlikeGecko',
                   'WindowsNT51rv310',
                   'WindowsNT60rv310',
                   'WindowsNT61AppleWebKit53736KHTMLlikeGecko',
                   'WindowsNT61WOW64AppleWebKit53736KHTMLlikeGecko',
                   'WindowsNT61WOW64Trident70rv110',
                   'WindowsNT61WOW64rv240',
                   'WindowsNT61WOW64rv300',
                   'WindowsNT61WOW64rv310',
                   'WindowsNT61WOW64rv320',
                   'WindowsNT61rv310',
                   'WindowsNT62WOW64AppleWebKit53736KHTMLlikeGecko',
                   'WindowsNT62WOW64rv310',
                   'WindowsNT63WOW64AppleWebKit53736KHTMLlikeGecko',
                   'WindowsNT63WOW64Trident70rv110',
                   'WindowsNT63WOW64rv310',
                   'X11Linuxx86_64AppleWebKit53736KHTMLlikeGecko',
                   'X11Linuxx86_64rv310',
                   'X11UbuntuLinuxi686rv310',
                   'X11UbuntuLinuxx86_64rv300',
                   'X11UbuntuLinuxx86_64rv310',
                   'compatibleMSIE100WindowsNT61WOW64Trident60',
                   'compatibleMSIE100WindowsNT62WOW64Trident60',
                   'compatibleMSIE90WindowsNT61Trident50',
                   'compatibleMSIE90WindowsNT61WOW64Trident50',
                   'iPadCPUOS7_1_1likeMacOSXAppleWebKit537512KHTMLlikeGecko',
                   'iPadCPUOS7_1_2likeMacOSXAppleWebKit537512KHTMLlikeGecko',
                   'iPhoneCPUiPhoneOS7_1_1likeMacOSXAppleWebKit537512KHTMLlikeGecko',
                   'iPhoneCPUiPhoneOS7_1_2likeMacOSXAppleWebKit537512KHTMLlikeGecko',
                   'iPhoneCPUiPhoneOS7_1likeMacOSXAppleWebKit537512KHTMLlikeGecko',
                   'Firefox240',
                   'Firefox300',
                   'Firefox310',
                   'Firefox320',
                   'Gecko',
                   'Safari5345910"',
                   'Safari53736"',
                   'Safari537749"',
                   'Safari5377514"',
                   'Safari537764"',
                   'Safari537774"',
                   'Safari537782"',
                   'Safari60013"',
                   'Safari60018"',
                   'Safari953753"',
                   'Trident50',
                   'Trident60',
                   'avg_session_events',
                   'min_session_events',
                   'max_session_events',
                   'total_session_events']

KNOWN_MESSAGES = ['About',
                  'Add Friend',
                  'Add to Playlist',
                  'Downgrade',
                  'Error',
                  'Help',
                  'Home',
                  'Logout',
                  'NextSong',
                  'Roll Advert',
                  'Save Settings',
                  'Settings',
                  'Submit Upgrade',
                  'Thumbs Down',
                  'Thumbs Up',
                  'Upgrade']


class UserLogTransformer(Transformer):
    """
    Custom transformer that aggregates Spotify log data into user features for predicting
    if they are at risk of churning.

    Extracted user information are:

    * user base data:
        * gender
        * level
        * user changed level in logged period
        * registration (time tamp)
    * aggregated log data
        * avg_session_events  mean number of log events per session
        * min_session_events
        * max_session_events
        * total_session_events
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
            .agg(F.first('registration').cast('double').alias('registration'),
                 ((F.max('ts')-F.min('ts')) / (3600 * 24 * 1000)).alias('period')
                 )
        # user browser and operating system counts
        df_os_browser = dataset.select('userId', 'userAgent',
                                       F.regexp_replace(F.regexp_extract(F.col('userAgent'), ".*?(\(.*\)).*", 1),
                                        '[\(\);:;\s\/.,]+', '').alias('os'),
                                       F.regexp_replace(F.regexp_extract(F.col('userAgent'), ".*\s(.*)", 1),
                                        '[\(\);:;\s\/.,]+', '').alias('browser'))
        df_os_onehot = df_os_browser.groupBy('userId').pivot('os', values=KNOWN_OS)\
            .agg(F.countDistinct('userId').cast('double').alias('os')).fillna(0)
        df_browser_onehot = df_os_browser.groupBy('userId').pivot('browser', values=KNOWN_BROWSERS) \
            .agg(F.countDistinct('userId').cast('double').alias('browser')).fillna(0)
        # user page counts and frequencies
        df_page_counts = dataset.select('userId', F.column('page').alias('page'))\
            .join(df_by_user.select('userId', 'period'), on='userId')\
            .groupBy('userId').pivot('page', values=KNOWN_MESSAGES).agg(F.count('userId').cast('double').
                                                                        alias('count'),
                                                                        (F.count('userId') /
                                                                         F.first('period')).cast('double')
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
            .groupBy('userId').agg(F.max('churn').alias('label'))


class TrainingAssembler(Transformer):
    """"
    Assemble the data set for training into label and feature vector columns.

    Expects a dataframe of numeric columns, one of which should be named 'churned'. The 'churned' column is used as the
    labels columns. All other columns are assembled into the features vector.
    """

    def _transform(self, dataset):
        input_cols = dataset.columns
        if 'label' in input_cols:
            input_cols.remove('label')
        vectassemble = VectorAssembler(inputCols=input_cols, outputCol='features', handleInvalid='skip')
        return vectassemble.transform(dataset).select('features', 'userId')


class MasterTransformer(Transformer):
    """
    Transformer instance that puts together the building blocks for label and feature extraction
    """

    def _transform(self, dataset):
        logtransform = UserLogTransformer()
        labeltransform = UserLabelTransformer()
        assembler = TrainingAssembler()
        return assembler.transform(logtransform.transform(dataset)).join(labeltransform.transform(dataset), on='userId')


class FeatureUnassembler(Transformer):
    """
    Disassemble a feature vector back to columns.

    For analysis of features stored from large scale ML runs.
    """

    def _transform(self, dataset):
        toarr = F.udf(lambda col: col.toArray().tolist(), ArrayType(DoubleType()))
        df_temp = dataset.withColumn("ar", toarr(F.col('features'))).select(['userId'] +
                                                                            [F.col("ar")[i] for i in
                                                                            range(len(FEATURE_COLUMNS))])
        return df_temp.toDF('userId', *FEATURE_COLUMNS)
