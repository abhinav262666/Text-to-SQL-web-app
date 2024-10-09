import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('your_database.db')

# Create a cursor object
cursor = conn.cursor()

# Fetch the list of all tables in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Print the list of tables
print("Tables in the database:")
for table in tables:
    print(table[0])

# For each table, query and print its content
for table in tables:
    table_name = table[0]
    print(f"\nContents of table '{table_name}':")
    
    # Fetch all rows from the current table
    cursor.execute(f"SELECT * FROM {table_name};")
    rows = cursor.fetchall()
    
    # Fetch column names for the current table
    cursor.execute(f"PRAGMA table_info({table_name});")
    column_info = cursor.fetchall()
    column_names = [col[1] for col in column_info]
    
    # Print column names
    print(f"Columns: {', '.join(column_names)}")
    
    # Print each row
    for row in rows:
        print(row)

# Close the connection
conn.close()
