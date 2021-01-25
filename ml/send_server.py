import psycopg2
import datetime

def send2server(data2send):
    connection = psycopg2.connect(user="blindtermite",
                                      password = "BlindAsAB@t",
                                      host="localhost",
                                      port="5444",
                                      database="termites")

    try:
       cursor = connection.cursor()

       postgres_insert_query = """ INSERT INTO termiteclusters (inserttime, co2,  pir, cluster, confidence, chipid) VALUES (%s,%s,%s,%s,%s,%s)"""
       cursor.execute(postgres_insert_query, data2send)

       connection.commit()
       count = cursor.rowcount
       print (count, "Record inserted successfully into mobile table")

    except (Exception, psycopg2.Error) as error :
        if(connection):
            print("Failed to insert record into termiteclusters table", error)

    finally:
        #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

#thetime = datetime.datetime.now()
#send2server((1, None , thetime , 400, 3, 12, 0.5, 867577))
