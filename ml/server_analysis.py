
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
   cursor.execute("SELECT inserttime, co2, pir, cluster, confidence, chipid FROM termiteclusters WHERE chipid = 8360978 AND inserttime <= '2021-05-17 10:34:11.903760'") # ORDER BY chipid
   rows = cursor.fetchall()
   print('Number of rows: ', cursor.rowcount)

   with open("test.txt", "w", newline='') as f:
        for row in rows:
            print(row, file=f)

   cursor.close()

except (Exception, psycopg2.Error) as error :
    if(connection):
        print("Failed to insert record into termiteclusters table", error)

finally:
    #closing database connection.
    if(connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
