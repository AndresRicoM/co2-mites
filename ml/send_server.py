import psycopg2
from config import config

def send2server(data2send):
    try:
       connection = psycopg2.connect(user="postgres",
                                      password="",
                                      host="127.0.0.1",
                                      port="5444",
                                      database="termites")
       cursor = connection.cursor()

       postgres_insert_query = """ NEW Data """
       cursor.execute(postgres_insert_query, data2send)

       connection.commit()
       count = cursor.rowcount
       print (count, "Record inserted successfully into mobile table")

    except (Exception, psycopg2.Error) as error :
        if(connection):
            print("Failed to insert record into mobile table", error)

    finally:
        #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
