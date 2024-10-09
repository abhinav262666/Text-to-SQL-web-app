import sqlite3

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('your_database.db')
print("Creating .........................")
# Create a cursor object
cursor = conn.cursor()

# Create a new table for storing metadata about each table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS table_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        datafrom TEXT,
        table_name TEXT,
        information TEXT
    )
''')

# Insert metadata about the SQL tables and the CSV table into the table_metadata
cursor.execute('''
    INSERT INTO table_metadata (datafrom, table_name, information)
    VALUES
    ('SQL', 'users', 'Contains user information such as user_id, session_id, name, and age.'),
    ('SQL', 'date_of_joining', 'Contains employee joining information: employee_id, name, and join_date.'),
    ('SQL', 'salary', 'Stores salary details for employees, including salary, bonus, and deduction.'),
    ('SQL', 'orders', 'Contains order details such as order_id, user_id, product_id, and quantity. References users and products tables.'),
    ('SQL', 'employees', 'Contains employee details: employee_id, name, department, and salary.'),
    ('SQL', 'departments', 'Stores department information such as department_id, department_name, and location.'),
    ('CSV', 'product_sales', 'Product sales data containing sale_id, product name, amount, and user_id.')
''')

# Commit the transaction
conn.commit()

# Close the connection
conn.close()

print("Metadata table created and populated successfully.")
