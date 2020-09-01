# Spark Churn
Application to predict whether a spotify user is likely to churn, i.e. to cancel or downgrade their subscription, 
based on their activity. User activity is read as a pre-processed web log in JSON format.

## Data Exploration
![Counts of log event types (log y-scale)](doc_img/Events_Histogram.png)
![Comparison of log event counts by churned and kept users (log y-scale)](doc_img/Event_Frequency_Compare.png)

![Histogram of users by number of their associated log events](doc_img/Interactions_per_user.png)
![Histogram of users by number of their associated log events for kept and churned users](doc_img/Interactions_per_user_compare.png)

## Operation

### Input Data Filtering
Most columns can be correctly typed upon reading, by providing schema information in DDL format to spark's JSON parser.
The exception to this is the userId column, which cannot be successfully parsed to any integer type by the JSON parser,
presumably due to the use of empty strings for raw entries without userId. The column is read as string instead and then
 cast to long. Time stamps are imported as long integer, as they are interpreted wrong by the import filter. For use as
 features for machine learning, this is not problematic, they can be used in integer encoding.

The input data contains entries with no associated user ID. Since these entries cannot be associated to a user, they
are useless for user behavior prediction and are therefore discarded.

Further pre-processing of the log data is not necessary.

### Feature Extraction
For prediction of user behavior, the log data is aggregated per user. A rich set of per-user features can be extracted 
easily by SQL-like operations:

* user base data:
    * gender
    * level - paid or unpaid subscription
    * registration (time tamp)
* aggregated event data
    * mean number of log events per session
    * minimum number of events per session
    * maximum number of events per session
    * total number of log entries for this user
    * timestamp of earliest log event
    * timestamp of most recent log event
    * period - interval in days between earliest and latest log event
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
        * Downgrade
        * Submit Downgrade
        * Submit Upgrade
    * frequencies of log event types defined as count / period for the events above
    * 1-hot encoded list of operating systems seen on userId
    * 1-hot encoded list of browser engines seen on userId
    
User first- and last name, as well as song names are not evaluated here. Song names would add a huge number of 
categorical variables and users outside the training set, on which the model would be used, are likely to have 
additional category levels.

Events interpreted as churn are deleted from the dataset:
* Cancellation Confirmation


Churn-related events that are deleted to avoid data leakage:
* Cancel

## Machine Learning Results

## Web Application

## Prerequisites
### Python Version
Requires Python version 3.8 or higher (tested using version 3.8.3)

### Libraries
* jupyter 1.0.0
* numpy 1.19.1
* pandas 1.1.0
* plotly 4.8.2
* pyspark 3.0.0
* spark 2.5.4
