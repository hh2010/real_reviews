"""
Yelp Business Rater
Uses NLP to Create a New, Simplified Rating (1 or 0)
"""

import time
import pickle
import psycopg2
import numpy

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

# Load the models we will be using
with open ('model/forest.pkl', 'rb') as f:
    forest = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Establish the PostgreSQL Connection
conn_string = "host=" + sql_ip + " dbname=" + sql_db + " user=" + sql_user + \
              " password=" + sql_pass
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

# Load the cursor with reviews
q = "DECLARE super_cursor CURSOR FOR SELECT id, review FROM yelp"
cur.execute(q)

# Grab one review at a time, rate it, and update column with ranking
start = time.time()

# while True:
cur.execute("FETCH 1000 FROM super_cursor")
rows = cur.fetchall()
ids, revs = ([x[0] for x in rows], [x[1] for x in rows])
nums = [numpy.random.randint(0,2) for x in range(1000)]

# if not rows:
#     break

# for i in rows:
v = vectorizer.transform(revs)
pred = forest.predict(v).tolist()
cur.executemany("UPDATE yelp SET rating = %s WHERE id = %s",
                 [(a, b) for a, b in zip(pred, ids)])

conn.commit()

# Print the elapsed time
end = time.time()
print(end - start)

# Close the connection
if conn:
    conn.close()