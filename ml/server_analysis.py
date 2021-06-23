
# -*- coding: UTF-8 -*-
#
#    ██████╗ ██████╗ ██████╗       ███╗   ███╗██╗████████╗███████╗███████╗
#   ██╔════╝██╔═══██╗╚════██╗      ████╗ ████║██║╚══██╔══╝██╔════╝██╔════╝
#   ██║     ██║   ██║ █████╔╝█████╗██╔████╔██║██║   ██║   █████╗  ███████╗
#   ██║     ██║   ██║██╔═══╝ ╚════╝██║╚██╔╝██║██║   ██║   ██╔══╝  ╚════██║
#   ╚██████╗╚██████╔╝███████╗      ██║ ╚═╝ ██║██║   ██║   ███████╗███████║
#    ╚═════╝ ╚═════╝ ╚══════╝      ╚═╝     ╚═╝╚═╝   ╚═╝   ╚══════╝╚══════╝
#
#   ╔═╗┬┌┬┐┬ ┬  ╔═╗┌─┐┬┌─┐┌┐┌┌─┐┌─┐       ╔╦╗╦╔╦╗  ╔╦╗┌─┐┌┬┐┬┌─┐  ╦  ┌─┐┌┐
#   ║  │ │ └┬┘  ╚═╗│  │├┤ ││││  ├┤   ───  ║║║║ ║   ║║║├┤  │││├─┤  ║  ├─┤├┴┐
#   ╚═╝┴ ┴  ┴   ╚═╝└─┘┴└─┘┘└┘└─┘└─┘       ╩ ╩╩ ╩   ╩ ╩└─┘─┴┘┴┴ ┴  ╩═╝┴ ┴└─┘
#
#                                   .|
#                                  | |
#                                  |'|            ._____
#                          ___    |  |            |.   |' .---"|
#                  _    .-'   '-. |  |     .--'|  ||   | _|    |
#               .-'|  _.|  |    ||   '-__  |   |  |    ||      |
#               |' | |.    |    ||       | |   |  |    ||      |
#            ___|  '-'     '    ""       '-'   '-.'    '`      |____
#
#

import time
import psycopg2
import datetime

connection = psycopg2.connect(user="blindtermite",
                                  password = "BlindAsAB@t",
                                  host="localhost",
                                  port="5444",
                                  database="termites")

try:
   cursor = connection.cursor()
   cursor.execute("SELECT inserttime, co2, pir, cluster, chipid FROM termiteclusters WHERE chipid = 8360978 AND inserttime <= datetime.datetime(2021-05-17)") # ORDER BY chipid
   rows = cursor.fetchall()
   print('Number of rows: ', cursor.rowcount)
   #for row in rows:
   #   print(row)
   with open("/", "w", newline='') as f:
        for row in cur:
            print(row[0], file=f)
   cursor.close()
   #postgres_insert_query = """ INSERT INTO termxiteclusters (inserttime, co2,  pir, cluster, confidence, chipid) VALUES (%s,%s,%s,%s,%s,%s)"""
   #cursor.execute(postgres_insert_query, data2send)

   #connection.commit()
   #count = cursor.rowcount
   #print (count, "Record inserted successfully into mobile table")

except (Exception, psycopg2.Error) as error :
    if(connection):
        print("Failed to insert record into termiteclusters table", error)

finally:
    #closing database connection.
    if(connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
