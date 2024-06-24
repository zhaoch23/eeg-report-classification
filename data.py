import sqlite3
import pandas as pd

def db_iterator(db_path):
    """
    Iterate over the database and yield each report
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = "SELECT `Hashed ID`, `Report`, `Norm(1)/No(0)` FROM reports"
    try:
        cursor.execute(query)
        # Fetch rows one by one
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield row
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        yield None
    finally:
        conn.close()

def fetch_reports_as_df(db_path, count=100):
    """
    Fetch all reports from the database and return as a DataFrame
    """
    conn = sqlite3.connect(db_path)
    query = f"SELECT `Hashed ID`, `Report`, `Norm(1)/No(0)` FROM reports LIMIT {count}"
    df = pd.read_sql_query(query, conn)
    df.rename(columns={"Hashed ID": "ID", "Report": "Text", "Norm(1)/No(0)": "Label"}, inplace=True)
    conn.close()
    return df

def clean_text(txt):
    """
    Clean the text that only keeps the description and the impression part
    """
    description_marker = "DESCRIPTION OF THE RECORD"
    description_start = txt.find(description_marker)
    if (description_start == -1):
        return txt
    description = txt[description_start:].strip()
    description = description.replace("\n\n", "\n") # Remove double newlines
    return description