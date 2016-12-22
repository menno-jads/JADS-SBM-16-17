#######
# JADS - SBM 16/17
# Authors Joep van Ganzewinkel and Menno van Leeuwen
#

# sqlalchemy used for connection to MySQL
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# Pandas and Numpy
import pandas as pd
import numpy as np

#Sentiment analyser
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


# Wrapper class for our application
class AppAnalyser():
    def __init__(self, inputFile):
        Base = declarative_base()
        metadata = MetaData()
        # This line would change based on the RDBMS, so check your own
        # database connector
        self.engine = create_engine('mysql+pymysql://root:sHz>v%d(,1I7@localhost/sbm')
        Base.metadata.create_all(self.engine)
        tables = metadata.reflect(self.engine)
        self.inspector = inspect(self.engine)
        # Display table info
        self.display_tables()
        # Get list of id's
        companies_of_interest = self.get_companies_of_interest(inputFile)
        # Insert this list in our "main" method which does almost everything.
        self.get_company_info(companies_of_interest)
        # Define outputpaths, may be changed to your preference
        outputRatingPath = 'results/' + inputFile + '_rating.csv'
        outputRatingSummaryPath = 'results/' + inputFile + '_summary_rating.csv'
        outputTextPath = 'results/' + inputFile + '_textanalysis.csv'
        outputTextSummaryPath = 'results/' + inputFile + '_summary_textanalysis.csv'
        # Finally store our results to csv files
        self.text_df.to_csv(outputTextPath, index=False, sep=',', encoding='utf-8')
        self.rating_df.to_csv(outputRatingPath, index=False, sep=',', encoding='utf-8')
        self.rating_summary_df.to_csv(outputRatingSummaryPath, index=False, sep=',', encoding='utf-8')
        self.text_summary_df.to_csv(outputTextSummaryPath, index=False, sep=',', encoding='utf-8')


    # Help method to show the tables of the database
    def display_tables(self):
        for table_name in self.inspector.get_table_names():
            print("Table: %s" % table_name)
            self.display_columns(table_name)
            print()
            print()
    # Help method to show the columns/attributes of the database
    def display_columns(self, table):
        for column in self.inspector.get_columns(table):
            print("Column: %s" % column['name'])

    # Make a list of id's from file so we can query on these specific id's
    def get_companies_of_interest(self, inputFile):
        idCSV = pd.read_csv(inputFile + '.csv')
        idString = str(idCSV['id'].tolist()).strip('[]')
        return idString

    # Method to get some comany info. and to get all info
    # This by
    def get_company_info(self, companies_of_interest):
        # Select query, were our list of company id's is dynamic.
        results = self.engine.execute('\
        SELECT id, title, week, watch, appversion \
        FROM app_details \
        WHERE id IN ({0})\
        AND week = "20" \
        ORDER BY title;'.format(companies_of_interest))
        # DF to store initial results.
        self.result_df = pd.DataFrame(results.fetchall(), columns=['id', 'title', 'week', 'watch_support', 'version'])
        # DF will be used results per week
        self.rating_df = pd.DataFrame(columns=["id", "title", "week", "nr_observations", "week_mean"])
        # DF will be used to store result pre and post launch
        self.rating_summary_df = pd.DataFrame(columns=["id", "title", "pre_or_post", "mean", "improvement"])
        # Same as rating for the above 2 but now for the text analysis.
        self.text_df = pd.DataFrame(columns=["appID", "title", "week", "pos", "neg", "pospercentage", "nr_observations"])
        self.text_summary_df = pd.DataFrame(columns=["id", "title", "pre_or_post", "mean_of_percentage", "improvement"])
        for index, item in self.result_df.iterrows():
            # Method to evaluate the user rating per app
            self.average_rating(index, item[0], item[1])
            # Method to evaluate the user reviews per app
            self.review_sentiment_analyser(index, item[0], item[1])

    def average_rating(self, index, appID, title):
        weekResult = []
        observationResult = []
        # Weeks to Evaluate 16-23, can be changed if you want to
        for week in range(16, 24):
            results = self.engine.execute('\
            SELECT rating \
            FROM app_reviews \
            WHERE appID={0} AND week={1}'.format(appID, week))
            df = pd.DataFrame(results.fetchall(), columns=['rating'])
            if df.empty: # if no results found store NaN
                week_mean = np.nan
                nr_observations = np.nan
            else:
                # Calculate mean
                week_mean = df['rating'].mean()
                # Get nr of observations, used as weighted mean for
                # aggregated mean
                nr_observations = df.shape[0]
            weekResult.append(week_mean)
            observationResult.append(nr_observations)
            self.rating_df.loc[self.rating_df.shape[0]] = [appID, title, week, nr_observations, week_mean]
        # Calculate result pre launch
        preMean = self.calculate_mean(weekResult[0:4], observationResult[0:4])
        # Calculate result post launch
        postMean = self.calculate_mean(weekResult[5:8], observationResult[5:8])
        # Boolean if a improvement is detected
        improvement = postMean > preMean
        # Store mean values in rating summary dataframe
        self.rating_summary_df.loc[self.rating_summary_df.shape[0]] = [appID, title, "pre", preMean, improvement]
        self.rating_summary_df.loc[self.rating_summary_df.shape[0]] = [appID, title, "post", postMean, improvement]

    # Calculate aggregated mean for both pre and post launch
    def calculate_mean(self, numberList, observations):
        # transform the means and observations to numpy arrays
        np_numberList = np.array(numberList)
        np_observations = np.array(observations)
        # select only those indeces that contain values.
        indices = np.where(np.logical_not(np.isnan(np_numberList)))[0]
        # if we found values calculate the weighted mean
        if indices.size is not 0:
            mean = np.average(np_numberList[indices], weights=np_observations[indices])
        else:
            mean = np.nan
        return mean

    # Calculate sentiment based on the VADER method
    def review_sentiment_analyser(self, index, appID, title):
        weekResult = []
        observationResult = []
        # Weeks to Evaluate 16-23, can be changed if you want to
        for week in range(16, 24):
            # Execute select string for reviews of that week
            results = self.engine.execute(' \
            SELECT content \
            FROM app_reviews \
            WHERE appID={0} AND week={1}'.format(appID, week))
            df = pd.DataFrame(results.fetchall(), columns=['content'])
            pos = np.nan
            neg = np.nan
            if df.empty:
                week_mean = np.nan
                nr_observations = np.nan
            else:
                #Create VADER instance (see #13 for import)
                sid = SIA()
                for indexx, row in df.iterrows():
                    # Calculatea compound score given the user review.
                    sentiment = sid.polarity_scores(row['content'])
                    if sentiment['compound'] > 0:
                        # is the results positive or negative
                        if np.isnan(pos):
                            pos = 0
                        else:
                            pos += 1
                    else:
                        if np.isnan(neg):
                            neg = 0
                        else:
                            neg += 1
            nr_observations = df.shape[0]
            if nr_observations > 0:
                pospercentage = pos/nr_observations
            else:
                pospercentage = np.nan
            weekResult.append(pospercentage)
            observationResult.append(nr_observations)
            self.text_df.loc[self.text_df.shape[0]] = [appID, title, week, pos, neg, pospercentage, nr_observations]
        preMean = self.calculate_mean(weekResult[0:4], observationResult[0:4])
        postMean = self.calculate_mean(weekResult[5:8], observationResult[5:8])
        improvement = postMean > preMean
        self.text_summary_df.loc[self.text_summary_df.shape[0]] = [appID, title, "pre", preMean, improvement]
        self.text_summary_df.loc[self.text_summary_df.shape[0]] = [appID, title, "post", postMean, improvement]

# create instance of AppAnalyser for the favored group and execute it
db_favored = AppAnalyser('favored')
# create instance of AppAnalyser for the competitors group and execute it
db_competitors = AppAnalyser('competitors')
