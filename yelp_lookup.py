#!/usr/bin/env python
"""
This script will load the models created in the
Real Reviews Jupyter notebook.
Specify a Business ID number to calculate
a new adjusted rating for the business.
"""

import time
import argparse
import pickle
import json

# Set up the Argument Parser to grab Business ID number
parser = argparse.ArgumentParser()
parser.add_argument('--bid', default='JyxHvtj-syke7m9rbza7mA')
args = parser.parse_args()

# Load the pickled models
with open('model/forest_100k_eng.pkl', 'rb') as f:
    forest = pickle.load(f)

with open('model/vectorizer_100k_eng.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the necessary business data from JSON file
#  d = []
reviews = []
stars = []

a = time.time()
with open('data/yelp_academic_dataset_review.json', 'r') as f:
    for line in f:
        if json.loads(line.strip())['business_id'] == args.bid:
            #  d.append(json.loads(line.strip()))
            reviews.append(json.loads(line.strip())['text'])
            stars.append(json.loads(line.strip())['stars'])

# Calculate average ranking
vect_rev = vectorizer.transform(reviews)
for_pred = forest.predict(vect_rev)
b = time.time()

# Output results
stars_avg = sum(map(float, stars))/len(for_pred)

print "Old Reviews"
print stars
print "Old Reviews (Mean):", stars_avg
print "Old Reviews (%):", stars_avg/5
print "New Reviews"
print for_pred
print "New Rating (%):", sum(map(float, for_pred))/len(for_pred)
print "Time:", (b - a)

# Testing
#  print d
