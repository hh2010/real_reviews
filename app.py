import realrev
from flask import Flask, render_template, request
from wtforms import Form

# Get secret key
key = ""
with open('data/pvt.csv') as f:
    for line in f:
        l = line.strip().split(',')
        if l[0] == 'key':
            key = l[1]

# Configure the Flask app
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = key

# Set up the default route
@app.route("/", methods=['GET', 'POST'])
def reviews():
    """
    Load the Real Reviews page and populate the variables used on the site
    """
    form = Form(request.form)
    bus_id = request.form.get("business")
    print bus_id
    try:
        www_dict = bus_data(bus_id)
    except:
        print "There was an error!"
        www_dict = {}
    return render_template('reviews.html', form=form, data=www_dict)


def bus_data(bus_id):
    """
    Populate business data from Real Reviews module and Yelp API
    Input: Business ID number
    Output: Dictionary
    Side-Effects: Updated SQL database
    """
    # Set up a default dictionary
    bus_dict = {'business_id': 'Business ID', 'name': 'Business Name',
                'city': 'City', 'country': 'Country', 'old_rating': 0,
                'new_rating': 0,  'rev_count': 0, 'count_5': 0, 'count_4': 0,
                'count_3': 0, 'count_2': 0, 'count_1': 0, 'fav_count': 0,
                'unfav_count': 0}

    # Instantiate the SQL class for the business data we will be pulling
    sql = realrev.Sql(bus_id)

    # Check if we have previously analyzed the requested business
    # If not, pull the raw data and processing the data
    if sql.check() is False:
        pass
    else:
        sql_tup = sql.check()[0]
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
        return bus_dict

    # Get business data (name, country, etc) from Yelp API
    # Limited to 25,000 Yelp API calls per day
    # There are over 4 million reviews and over 140,000 businesses in database
    bus_info = realrev.Yelp(bus_id)

    #  Grab review text from SQL database
    sql.pull_reviews()

    # Use our trained Random Forests model and TFIDF vectorizer
    #  to determine whether each review is "Favorable" or "Unfavorable"
    realrev.model.predict(sql.reviews)

    # Conduct sentiment analysis and evaluate word counts in order to
    #  "penalize" the weighting of reviews that don't fit the threshold
    nlp = realrev.Nlp(sql.reviews, sql.stars, realrev.model.preds,
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
    fav_count = (realrev.model.preds == 1).sum()
    unfav_count = (realrev.model.preds == 0).sum()
    bus_tup = (bus_id, name, city, country, old_rating, new_rating,
               rev_count, count_5, count_4, count_3, count_2, count_1,
               fav_count, unfav_count)
    sql.insert(bus_tup)
    for key in bus_dict.keys():
        x = 0
        bus_dict[key] = bus_tup[x]
        x += 1
    return bus_dict

if __name__ == "__main__":
    app.run()
