from dotenv import load_dotenv
import psycopg2
import os

load_dotenv()

def get_db_cursor():
    conn = psycopg2.connect(
        # ====== VisionChat =======
        host= os.getenv("HOST"),
        user= os.getenv("USER"),
        password=os.getenv("PASSWORD"),
        dbname=os.getenv("DBNAME"),
        port=5432
    )
    return conn, conn.cursor()
