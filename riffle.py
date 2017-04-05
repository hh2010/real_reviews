#!/usr/bin/env python
# encoding=utf8
"""
This script will load the models created in the
Riffle Jupyter notebook.
Specify a Business ID number to calculate
a new adjusted rating for the business.
"""

# Add function to compare old score to new (which ones flipped)
# Can we look up and compare the text of the reviews?

import os
import sys
import time
import argparse
import pickle
import psycopg2
import xgboost as xgb
import numpy as np
from textblob import TextBlob
from yelpapi import YelpAPI

# Set current working directory
cwd = os.path.dirname(__file__)

# Set default encoding
reload(sys)
sys.setdefaultencoding('utf8')

# Set up the Argument Parser to grab Business ID number
parser = argparse.ArgumentParser()
parser.add_argument('--bus_id', default='JyxHvtj-syke7m9rbza7mA')
parser.add_argument('--input', default=os.path.join(cwd, 'data/bus_id.csv'))
args = parser.parse_args()

# Create our classes to be used in this file or exported to Flask app


class Sql(object):
    """
    Connection to Postgres server
    """
    def __init__(self, bus_id=args.bus_id):
        self.bus_id = bus_id
        self.sql_user = ""
        self.sql_pass = ""
        self.sql_ip = ""
        self.sql_db = ""
        self.cur = ""
        self.query = ""
        self.tups = ()
        self.names = ()
        self.reviews = []
        self.stars = []
        self.conn = ""

        with open(os.path.join(cwd, 'data/pvt.csv')) as f:
            for line in f:
                l = line.strip().split(',')
                if l[0] == 'sql_user':
                    self.sql_user = l[1]
                    continue
                if l[0] == 'sql_pass':
                    self.sql_pass = l[1]
                    continue
                if l[0] == 'sql_ip':
                    self.sql_ip = l[1]
                    continue
                if l[0] == 'sql_db':
                    self.sql_db = l[1]
                    continue

        self.conn_string = "host=" + self.sql_ip + " dbname=" + self.sql_db + \
                           " user=" + self.sql_user + " password=" + \
                           self.sql_pass

    def check(self):
        """
        Checks to see if review data exists in stored database
          so we don't have to waste time pulling data and modeling
        Input: None
        Output: Tuple of SQL output with business data
        Side-Effects: None
        """
        self.conn = psycopg2.connect(self.conn_string)
        self.cur = self.conn.cursor()
        self.cur.execute("""
                         SELECT * FROM yelp_stored WHERE business_id = %s;
                         """, (self.bus_id,))
        sql_tup = self.cur.fetchall()
        self.conn.close()
        if sql_tup == []:
            return False
        else:
            return sql_tup

    def insert(self, bus_tup):
        """
        Update the SQL database with business data
        Input: Tuple of Business Data to be Inserted into SQL
        Output: None
        Side-Effects: Updated SQL database
        """
        self.conn = psycopg2.connect(self.conn_string)
        self.cur = self.conn.cursor()
        self.query = """
                INSERT INTO yelp_stored (business_id, name, city, country,
                old_rating, new_rating, rev_count, count_5, count_4, count_3,
                count_2, count_1, fav_count, unfav_count, avg_wts)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s)
                """
        self.cur.execute(self.query, bus_tup)
        self.conn.commit()
        self.conn.close()

    def pull_reviews(self):
        """
        Input: Yelp Business ID
        Output: None
        Side-Effects: Update list of Yelp review text and star ratings
        """
        self.conn = psycopg2.connect(self.conn_string)
        self.cur = self.conn.cursor()
        self.cur.execute("""
                         SELECT review, stars FROM yelp WHERE business_id = %s;
                         """, (self.bus_id,))
        self.tups = self.cur.fetchall()
        self.conn.close()

        for rev, star in self.tups:
            self.reviews.append(rev)
            self.stars.append(star)

    def pull_names(self):
        """
        Input: None
        Output: Tuple of Business Names and IDs from the database
        Side-Effects: None
        """
        # Note: This only pulls Las Vegas businesses since that's what we are
        #        focusing on for demo
        self.conn = psycopg2.connect(self.conn_string)
        self.cur = self.conn.cursor()
        self.cur.execute("""
                         SELECT DISTINCT on (name) name, business_id FROM yelp_stored
                         WHERE city = 'Las Vegas' ORDER BY name asc;
                         """)
        self.names = self.cur.fetchall()
        self.conn.close()


class Yelp(object):
    """
    Connect to Yelp API and pull information
    """
    def __init__(self, bus_id=args.bus_id):
        self.bus_id = bus_id
        self.yelp_id = ""
        self.yelp_secret = ""
        self.response = {}
        self.name = ""
        self.city = ""
        self.country = ""

        with open(os.path.join(cwd, 'data/pvt.csv')) as f:
            for line in f:
                l = line.strip().split(',')
                if l[0] == "yelp_id":
                    self.yelp_id = l[1]
                    continue
                if l[0] == "yelp_secret":
                    self.yelp_secret = l[1]
                    continue

        self.yelp_api = YelpAPI(self.yelp_id, self.yelp_secret)
        self.pull_info()

    def pull_info(self):
        """
        Input: Yelp Business ID
        Output: None
        Side-Effects: Create city, country and name attributes for our business
        """
        self.response = self.yelp_api.business_query(self.bus_id)
        self.name = self.response['name']
        self.city = self.response['location']['city']
        self.country = self.response['location']['country']


class Model(object):
    """
    Load the sklearn review evaluatoin models
    """
    def __init__(self):
        self.vect_rev = []
        self.dmat = []
        self.probs = []
        self.preds = []

        # with open('model/forest_200k_eng.pkl', 'rb') as f:
        #     self.forest = pickle.load(f)

        with open(os.path.join(cwd, 'model/vectorizer_200k_eng.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)

        self.bst = xgb.Booster({'nthread': 4})
        self.bst.load_model(os.path.join(cwd, 'model/bst_200k.model'))

    def predict(self, reviews):
        """
        Make Favorable (1) vs Unfavorable (0) ratings for each
          review in the input list of reviews
        Input: List of Yelp reviews
        Output: None
        Side-Effects: Create list of new review ratings
        """
        self.vect_rev = self.vectorizer.transform(reviews)
        self.dmat = xgb.DMatrix(self.vect_rev)
        self.probs = self.bst.predict(self.dmat)

        # Get tough on reviews by requiring 0.6 probability threshold
        self.preds = 1 * (self.probs > 0.6)


class Nlp(object):
    """
    Calculate polarity and subjectivity, and word counts
     and apply weightings to the reviews accordingly
    """
    def __init__(self, reviews, stars, preds, name, country, city,
                 pols_bar=(0.6, -0.6), subj_bar=0.6, wc_bar=35, pols_pen=0.65,
                 subj_pen=0.65, wc_pen=0.25):

        """
        Penalize reviews based on sentiment analysis and word count
        Input: Thresholds for polarity and subjectivity
               Penalty amounts for each threshold
        Outputs: None
        Side-Effects: Create lists of penalties for reviews
        """

        # Assign the attributes from Yelp class
        self.name = name
        self.country = country
        self.city = city

        # Assign the attributes that will be used for examples of
        #  reviews that have "changed" due to the model as per the
        #  ('examples' function) below

        self.reviews = []
        self.stars = []
        self.revs_2 = []
        self.revs_3 = []
        self.revs_4 = []

        # Calculate the penalties
        self.sentiments = [TextBlob(i).sentiment for i in reviews]
        self.pols = [i[0] for i in self.sentiments]
        self.subj = [i[1] for i in self.sentiments]
        self.wc = [len(x.split()) for x in reviews]

        self.pols_wts = [pols_pen if any([x >= pols_bar[0], x <= pols_bar[1]]) else 1 for x in self.pols]
        self.subj_wts = [subj_pen if x >= subj_bar else 1 for x in self.subj]
        self.wc_wts = [wc_pen if x <= wc_bar else 1 for x in self.wc]
        self.avg_wts_tmp = [np.mean(x) for x in zip(self.pols_wts,
                                                    self.subj_wts)]
        self.avg_wts = [x if x != 1 else y for x, y in zip(self.wc_wts,
                                                           self.avg_wts_tmp)]
        self.pred_wts = [i*j for i, j in zip(preds, self.avg_wts)]
        self.stars_avg = sum(map(float, stars))/len(stars)
        self.new_rating = sum(map(float, self.pred_wts))/float(sum(self.avg_wts))

    def examples(self, reviews, stars, probs):
        """
        Determine which 2-4 star reviews had changed from positive
            to negative or vice versa
        Input: List of review text(from SQL) and corresponding Yelp stars
        Output: None
        Side-Effects: Create lists of reviews and their corresponding XGBoost
                        prediction probability)
        """
        self.reviews = reviews
        self.stars = stars

        for i in range(len(self.stars)):
            if stars[i] == 2:
                self.revs_2.append([self.reviews[i], probs[i]])
            if stars[i] == 3:
                self.revs_3.append([self.reviews[i], probs[i]])
            if stars[i] == 4:
                self.revs_4.append([self.reviews[i], probs[i]])

    def output(self):
        """
        Calculate the new ratings and output the results
        """
        print "Name:", self.name
        print "City:", self.city
        print "Country:", self.country
        print "Number of Reviews:", len(self.sentiments)
        print "Old Reviews (Stars):", self.stars_avg
        print "Old Reviews (%):", self.stars_avg/5
        print "New Rating (Stars)", self.new_rating*5
        print "New Rating (%):", self.new_rating


# Script to test results
# Load the model before running function (most time consuming)

model = Model()


def run(bus_id=args.bus_id):
    """ Run a Business ID through the model and output some results"""
    start_sql = time.time()
    sql = Sql(bus_id)
    end_sql = time.time()
    start_info = time.time()
    try:
        bus_info = Yelp(sql.bus_id)
        end_info = time.time()
        start_model = time.time()
    except:
        print "Business ID Does Not Exist! Exiting..."
        return
    sql.pull_reviews()
    model.predict(sql.reviews)
    end_model = time.time()
    start_nlp = time.time()
    nlp = Nlp(sql.reviews, sql.stars, model.preds, bus_info.name,
              bus_info.country, bus_info.city)
    nlp.examples(sql.reviews, sql.stars, model.probs)
    end_nlp = time.time()
    print "SQL Time: ", (end_sql - start_sql)
    print "Info Time: ", (end_info - start_info)
    print "Model Time: ", (end_model - start_model)
    print "NLP Time: ", (end_nlp - start_nlp)


def update_db(bus_id=args.bus_id):
    """ Run a Business ID through the model and insert into SQL """
    # Instantiate the SQL class for the business data we will be pulling
    sql = Sql(bus_id)

    # Check if we have previously analyzed the requested business
    # If not, pull the raw data and processing the data
    if sql.check() is not False:
        print "Already in database!"
        return

    # Get business data (name, country, etc) from Yelp API
    # Limited to 25,000 Yelp API calls per day
    # There are over 4 million reviews and over 140,000 businesses in database
    while True:
        try:
            bus_info = Yelp(bus_id)
            break
        except ValueError:
            pass
        except YelpAPI.YelpAPIError:
            return

    #  Grab review text from SQL database
    sql.pull_reviews()

    # Use our trained XGBoost Classifier and TFIDF vectorizer
    #  to determine whether each review is "Favorable" or "Unfavorable"
    model.predict(sql.reviews)

    # Conduct sentiment analysis and evaluate word counts in order to
    #  "penalize" the weighting of reviews that don't fit the threshold
    nlp = Nlp(sql.reviews, sql.stars, model.preds,
              bus_info.name, bus_info.country, bus_info.city)

    # Assign variables from all the objects attributes we created
    #  and then input them into a tuple.
    # The tuple is used to populate the SQL database for faster lookup of
    #  our analysis at a later time
    # The tuple is also used to populate our dictionary which will be
    #  used for variables that will be rendered on our website
    name = nlp.name
    city = nlp.city
    country = nlp.country
    old_rating = int(100 * nlp.stars_avg / 5)
    new_rating = int(nlp.new_rating * 100)
    rev_count = len(sql.reviews)
    count_5 = sql.stars.count(5)
    count_4 = sql.stars.count(4)
    count_3 = sql.stars.count(3)
    count_2 = sql.stars.count(2)
    count_1 = sql.stars.count(1)
    fav_count = (model.preds == 1).sum()
    unfav_count = (model.preds == 0).sum()
    avg_wts = int(100*sum(nlp.avg_wts) / len(nlp.avg_wts))
    bus_tup = (bus_id, name, city, country, old_rating, new_rating,
               rev_count, count_5, count_4, count_3, count_2, count_1,
               fav_count, unfav_count, avg_wts)
    sql.insert(bus_tup)
    print bus_tup
