"""
Create the table of Yelp reviews in SQL database
Load a trained ratings model populate new ratings column in DB
"""

import time
import psycopg2
import sys

# Grab authentication info from file
sql_user = ""
sql_pass = ""
sql_ip = ""
sql_db = ""

with open('pvt.csv') as f:
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

# Establish the PostgreSQL Connection
conn_string = "host=" + sql_ip + " dbname=" + sql_db + " user=" + sql_user + \
              " password=" + sql_pass
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

# Confirm with user to proceed dropping table
ans = raw_input("This will drop table yelp2.  Are you sure? (Y/N)").lower()
if ans == 'y':
    break
else:
    print "Exiting..."
    sys.exit()

# Create the Yelp table on Postgres server
cur.execute("DROP TABLE IF EXISTS yelp2;")
cur.execute("""
            CREATE TABLE yelp2(id SERIAL PRIMARY KEY, review_id TEXT,
            user_id TEXT, business_id TEXT, dt DATE, review TEXT, useful SMALLINT,
            funny SMALLINT, cool SMALLINT, stars SMALLINT, rating SMALLINT);
            """)
conn.commit()

# Write the entire Yelp Review dataset to database
#  leaving placeholders for new ratings
start = time.time()
with open('yelp_academic_dataset_review.json') as f:
    for line in range(100):
        d = json.loads(f.readline().strip())
        ds = [d['review_id'], d['user_id'], d['business_id'], d['date'],
              d['text'], d['useful'], d['funny'], d['cool'], d['stars'], 0]
        query = """
                INSERT INTO yelp2 (review_id, user_id, business_id,
                dt, review, useful, funny, cool, stars, rating)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
        cur.execute(query, ds)
        conn.commit()

# Print total time elapsed
end = time.time()
print(end-start)

# Close the connection
if conn:
    conn.close()