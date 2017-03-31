#!/usr/bin/env python
# encoding=utf8
"""
This script will load the models created in the
Real Reviews Jupyter notebook.
Specify a Business ID number to calculate
a new adjusted rating for the business.
"""

import sys
import time
import argparse
import pickle
import json
import psycopg2
import numpy as np
from textblob import TextBlob
from yelpapi import YelpAPI

# Set default encoding
reload(sys)
sys.setdefaultencoding('utf8')

# Grab SQL and Yelp authentication info from file
sql_user = ""
sql_pass = ""
sql_ip = ""
sql_db = ""
yelp_id = ""
yelp_secret = ""

with open('data/pvt.csv') as f:
    for line in f:
        l = line.strip().split(',')
        if l[0] == 'sql_user':
            sql_user = l[1]
            continue
        if l[0] == 'sql_pass':
            sql_pass = l[1]
            continue
        if l[0] == 'sql_ip':
            sql_ip = l[1]
            continue
        if l[0] == 'sql_db':
            sql_db = l[1]
            continue
        if l[0] == "yelp_id":
            yelp_id = l[1]
            continue
        if l[0] == "yelp_secret":
            yelp_secret = l[1]
            continue

# Set up the Argument Parser to grab Business ID number
parser = argparse.ArgumentParser()
parser.add_argument('--bid', default='JyxHvtj-syke7m9rbza7mA')
args = parser.parse_args()

# Load the pickled models
with open('model/forest_100k_eng.pkl', 'rb') as f:
    forest = pickle.load(f)

with open('model/vectorizer_100k_eng.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Establish the PostgreSQL Connection
conn_string = "host=" + sql_ip + " dbname=" + sql_db + " user=" + sql_user + \
              " password=" + sql_pass
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

# Pull reviews data from SQL
reviews = []
stars = []
cur.execute("SELECT review, stars FROM yelp WHERE business_id = %s;", (args.bid,))
tups = cur.fetchall()
conn.close()

for rev, star in tups:
    reviews.append(rev)
    stars.append(star)

# Pull Yelp API Business Info
yelp_api = YelpAPI(yelp_id, yelp_secret)
response = yelp_api.business_query(args.bid)
city = response['location']['city']
country = response['location']['country']
name = response['name']

# Load the necessary business data from JSON file
# reviews = []
# stars = []
#
start = time.time()
#
# with open('data/yelp_academic_dataset_review.json', 'r') as f:
#     for line in f:
#         if json.loads(line.strip())['business_id'] == args.bid:
#             reviews.append(json.loads(line.strip())['text'])
#             stars.append(json.loads(line.strip())['stars'])

# Calculate average ranking
vect_rev = vectorizer.transform(reviews)
for_pred = forest.predict(vect_rev)

end = time.time()

# Calculate polarity and subjectivity, and word counts
#  and apply weightings to the reviews accordingly

# These are the thresholds for which we will penalize reviews
pols_bar = [0.6, -0.6]
subj_bar = 0.6
wc_bar = 20

# This is the penalty we will apply
pols_pen = 0.75
subj_pen = 0.75
wc_pen = 0.25

# Calculate the penalties
sentiments = [TextBlob(i).sentiment for i in reviews]
pols = [i[0] for i in sentiments]
subj = [i[1] for i in sentiments]
wc = [len(x.split()) for x in reviews]

pols_wts = [pols_pen for x in pols if any([x >= pols_bar[0], x <= pols_bar[1]])]
subj_wts = [subj_pen for x in subj if x >= subj_bar]
wc_wts = [wc_pen for x in wc if x <= wc_bar]
avg_wts = [np.mean(x) for x in zip(pols_wts, subj_wts, wc_wts)]

pred_wts = [i*j for i, j in zip(for_pred, avg_wts)]

# Output results
stars_avg = sum(map(float, stars))/len(stars)

print "Name:", name
print "City:", city
print "Country:", country
print "Number of Reviews:", len(stars)
print "Old Reviews (Stars):", stars_avg
print "Old Reviews (%):", stars_avg/5
print "New Rating (%):", sum(map(float, pred_wts))/sum(avg_wts)
print "Time:", (end - start)
