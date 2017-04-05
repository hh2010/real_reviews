"""
Flask App to render the Riffle website
"""
# Warriors game?
import os
import riffle
from flask import Flask, render_template, request
# from flask_socketio import SocketIO, emit
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

# Configure the Flask app
# DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = key
# socketio = SocketIO(app)

# # Create function for Session Counter
# def sumSessionCounter():
#   try:
#     session['counter'] += 1
#   except KeyError:
#     session['counter'] = 1

# Set up the default route
@app.route("/", methods=['GET', 'POST'])
def reviews(names=bus_name_list, ids=bus_id_list):
    """
    Load the Riffle page and populate the variables used on the site
    """
    # # Initialize the counter, or increment it
    # sumSessionCounter()

    # Instantiate the bus_data class
    if selected:
        pass
    else:
        selected = bus_data()

    # Pull business name from the form and lookup the corresponding ID
    form = Form(request.form)

    try:
        bus_name = request.form.get("business")
        bus_id = ids[names.index(bus_name)]
    except:
        bus_name = 'Bellagio Hotel'
        bus_id = ids[names.index(bus_name)]

    # Pull the list of Business Names and IDs
    try:
        if vis % 2 == 0:
            www_dict2 = dict(selected.www_dict1)
            www_dict1 = selected.pull(bus_id)
        else:
            www_dict2 = dict(selected.www_dict1)
            www_dict1 = selected.pull(bus_id)
    except:
        print "There was an error!"
        selected = bus_data()

    return render_template('reviews.html', form=form, data1=www_dict1,
                           data2=www_dict2, bus_names=names,
                           default_bus=bus_name, vis=vis)


class bus_data(object):
    def __init__(self):
        # Set the table visibility counter to 0
        self.vis = 0

        # Set up the initial business dictionaries
        self.def_dict = {'business_id': 'Business ID', 'name': 'Business Name',
                         'city': 'City', 'country': 'Country', 'old_rating': 0,
                         'new_rating': 0, 'rev_count': 0, 'count_5': 0,
                         'count_4': 0, 'count_3': 0, 'count_2': 0,
                         'count_1': 0, 'fav_count': 0, 'unfav_count': 0,
                         'avg_wts': 0}
        self.bus_dict = dict(self.def_dict)
        self.www_dict1 = dict(self.def_dict)
        self.www_dict2 = dict(self.def_dict)

        # SQL tuple for pulling data from postgres
        self.sql_tup = ()

        # Empty variables for the Riffle instances we will create
        self.bus_id = ""
        self.sql = ""
        self.bus_info = ""
        self.nlp = ""

    def pull(self, bus_id):
        """
        Populate business data from Riffle module and Yelp API
        Input: Business ID number
        Output: Dictionary of the selected business data
        Side-Effects: Updated SQL database, updated www_dict attribute
        """
        # Create attribute for bus_id
        self.bus_id = bus_id

        # Reset the business data dictionary
        self.bus_dict = dict(self.def_dict)

        # Instantiate the SQL class for the business data we will be pulling
        self.sql = riffle.Sql(bus_id)

        # Check if we have previously analyzed the requested business
        # If not, pull the raw data and processing the data
        if self.sql.check() is False:
            pass
        else:
            self.sql_tup = self.sql.check()[0]

            # If we render a form for a new business, then increase
            #  the visibility variable to allow viewing of multiple
            #  business ratings
            if self.bus_dict['name'] == self.sql_tup[1]:
                pass
            else:
                self.vis += 1

            self.bus_dict['business_id'] = self.sql_tup[0]
            self.bus_dict['name'] = self.sql_tup[1]
            self.bus_dict['city'] = self.sql_tup[2]
            self.bus_dict['country'] = self.sql_tup[3]
            self.bus_dict['old_rating'] = self.sql_tup[4]
            self.bus_dict['new_rating'] = self.sql_tup[5]
            self.bus_dict['rev_count'] = self.sql_tup[6]
            self.bus_dict['count_5'] = self.sql_tup[7]
            self.bus_dict['count_4'] = self.sql_tup[8]
            self.bus_dict['count_3'] = self.sql_tup[9]
            self.bus_dict['count_2'] = self.sql_tup[10]
            self.bus_dict['count_1'] = self.sql_tup[11]
            self.bus_dict['fav_count'] = self.sql_tup[12]
            self.bus_dict['unfav_count'] = self.sql_tup[12]
            self.bus_dict['avg_wts'] = self.sql_tup[13]
            return self.bus_dict

        # Get business data (name, country, etc) from Yelp API
        # Limited to 25,000 Yelp API calls per day
        # There are over 4 million reviews and over 140,000 businesses in database
        self.bus_info = riffle.Yelp(bus_id)

        #  Grab review text from SQL database
        self.sql.pull_reviews()

        # Use our trained Random Forests model and TFIDF vectorizer
        #  to determine whether each review is "Favorable" or "Unfavorable"
        riffle.model.predict(self.sql.reviews)

        # Conduct sentiment analysis and evaluate word counts in order to
        #  "penalize" the weighting of reviews that don't fit the threshold
        self.nlp = riffle.Nlp(self.sql.reviews, self.sql.stars, riffle.model.preds,
                              self.bus_info.name, self.bus_info.country,
                              self.bus_info.city)

        # Assign variables from all the objects attributes we created
        #  and then input them into a tuple.
        # The tuple is used to populate the SQL database for faster lookup of
        #  our analysis at a later time
        # The tuple is also used to populate our dictionary which will be
        #  used for variables that will be rendered on our website
        name = self.nlp.name
        city = self.nlp.city
        country = self.nlp.country
        old_rating = int(100 * self.nlp.stars_avg / 5)
        new_rating = int(self.nlp.new_rating * 100)
        rev_count = len(self.sql.reviews)
        count_5 = self.sql.stars.count(5)
        count_4 = self.sql.stars.count(4)
        count_3 = self.sql.stars.count(3)
        count_2 = self.sql.stars.count(2)
        count_1 = self.sql.stars.count(1)
        fav_count = (riffle.model.preds == 1).sum()
        unfav_count = (riffle.model.preds == 0).sum()
        avg_wts = self.nlp.avg_wts.sum() / len(self.nlp.avg_wts)
        bus_tup = (bus_id, name, city, country, old_rating, new_rating,
                   rev_count, count_5, count_4, count_3, count_2, count_1,
                   fav_count, unfav_count, avg_wts)
        self.sql.insert(bus_tup)
        for item in self.bus_dict.keys()[:-1]:
            i = 0
            self.bus_dict[item] = bus_tup[x]
            i += 1
        self.vis += 1
        self.www_dict1 = dict(self.bus_dict)
        return self.www_dict1

# # Disconnect the session on close
#
# @socketio.on('disconnect')
# def disconnect():
#     session.pop('counter', None)

if __name__ == "__main__":
    app.run()