"""
Flask App to render the Riffle website
"""
import os
import riffle
from flask import Flask, render_template, request, session
from wtforms import Form

# Set current working directory
cwd = os.path.dirname(__file__)

# Get secret key from private CSV file
key = ""
with open(os.path.join(cwd, 'data/pvt.csv')) as f:
    for line in f:
        l = line.strip().split(',')
        if l[0] == 'key':
            key = l[1]

# Set up our list of Las Vegas Business IDs
sql = riffle.Sql()
sql.pull_names()
bus_name_list = [x for x, y in sql.names]
bus_id_list = [y for x, y in sql.names]

# Set up a default dictionary
def_dict = {'business_id': 'Business ID', 'name': 'Business Name',
            'city': 'City', 'country': 'Country', 'old_rating': 0,
            'new_rating': 0, 'rev_count': 0, 'count_5': 0, 'count_4': 0,
            'count_3': 0, 'count_2': 0, 'count_1': 0, 'fav_count': 0,
            'unfav_count': 0, 'avg_wts': 0}

# Initialize session variables
def session_init():
    try:
        if session['vis']:
            pass
        else:
            session['vis'] = 0
    except KeyError:
        session['vis'] = 0
    try:
        if session['www_dict1']:
            pass
        else:
            session['www_dict1'] = dict(def_dict)
    except KeyError:
        session['www_dict1'] = dict(def_dict)
    try:
        if session['www_dict2']:
            pass
        else:
            session['www_dict2'] = dict(def_dict)
    except KeyError:
        session['www_dict2'] = dict(def_dict)

# Configure the Flask app
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = key

# Set up the default route
@app.route("/", methods=['GET', 'POST'])
def reviews(names=bus_name_list, ids=bus_id_list):
    """
    Load the Riffle page and populate the variables used on the site
    """
    global def_dict
    session_init()
    form = Form(request.form)

    try:
        bus_name = request.form.get("business")
        bus_id = ids[names.index(bus_name)]
    except:
        bus_name = 'Bellagio Hotel'
        bus_id = ids[names.index(bus_name)]

    # Pull the list of Business Names and IDs
    try:
        session['www_dict2'] = dict(session['www_dict1'])
        session['www_dict1'] = bus_data(bus_id)
    except:
        print "There was an error!"
        session['vis'] = 0

    return render_template('reviews.html', form=form, data1=session['www_dict1'],
                           data2=session['www_dict2'], bus_names=names,
                           default_bus=bus_name, vis=session['vis'])

def bus_data(bus_id):
    """
    Populate business data from Riffle module and Yelp API
    Input: Business ID number
    Output: Dictionary
    Side-Effects: Updated SQL database
    """
    # global vis
    global def_dict

    bus_dict = dict(def_dict)

    # Instantiate the SQL class for the business data we will be pulling
    sql = riffle.Sql(bus_id)

    # Check if we have previously analyzed the requested business
    # If not, pull the raw data and processing the data
    if sql.check() is False:
        pass
    else:
        sql_tup = sql.check()[0]

        # If we render a form for a new business, then increase
        #  the visibility variable to allow viewing of multiple
        #  business ratings
        if bus_dict['name'] == sql_tup[1]:
            pass
        else:
            session['vis'] += 1

        bus_dict['business_id'] = sql_tup[0]
        bus_dict['name'] = sql_tup[1]
        bus_dict['city'] = sql_tup[2]
        bus_dict['country'] = sql_tup[3]
        bus_dict['old_rating'] = sql_tup[4]
        bus_dict['new_rating'] = sql_tup[5]
        bus_dict['rev_count'] = sql_tup[6]
        bus_dict['count_5'] = sql_tup[7]
        bus_dict['count_4'] = sql_tup[8]
        bus_dict['count_3'] = sql_tup[9]
        bus_dict['count_2'] = sql_tup[10]
        bus_dict['count_1'] = sql_tup[11]
        bus_dict['fav_count'] = sql_tup[12]
        bus_dict['unfav_count'] = sql_tup[12]
        bus_dict['avg_wts'] = sql_tup[13]
        return bus_dict

    # Instantiate the XGBoost model
    model = riffle.Model()

    # Get business data (name, country, etc) from Yelp API
    # Limited to 25,000 Yelp API calls per day
    # There are over 4 million reviews and over 140,000 businesses in database
    bus_info = riffle.Yelp(bus_id)

    #  Grab review text from SQL database
    sql.pull_reviews()

    # Use our trained Random Forests model and TFIDF vectorizer
    #  to determine whether each review is "Favorable" or "Unfavorable"
    model.predict(sql.reviews)

    # Conduct sentiment analysis and evaluate word counts in order to
    #  "penalize" the weighting of reviews that don't fit the threshold
    nlp = riffle.Nlp(sql.reviews, sql.stars, model.preds,
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
    avg_wts = nlp.avg_wts.sum() / len(avg_wts)
    bus_tup = (bus_id, name, city, country, old_rating, new_rating,
               rev_count, count_5, count_4, count_3, count_2, count_1,
               fav_count, unfav_count, avg_wts)
    sql.insert(bus_tup)
    for key in bus_dict.keys()[:-1]:
        x = 0
        bus_dict[key] = bus_tup[x]
        x += 1
    session['vis'] += 1
    return bus_dict

if __name__ == "__main__":
    app.run()
