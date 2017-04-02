#!/usr/bin/env python
"""
This is a script to iterate through a list of Business ID
numbers and populate the SQL database

Input files must be in the CSV format "#reviews,business_id"
  Example: "997,C8D_GU9cDDjbOJfCaGXxDQ"
"""

import sys
import realrev

# Set the default encoding
reload(sys)
sys.setdefaultencoding('utf8')

# Go through every Business ID in the file and update the SQL database
with open(realrev.args.input, 'r') as f:
    for line in f:
        bus_id = line.strip().split(',')[1]
        realrev.update_db(bus_id)