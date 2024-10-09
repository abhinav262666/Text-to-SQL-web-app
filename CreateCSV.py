import pandas as pd

# Create a dictionary with some sample data
data = {
    'sale_id': [1, 2, 3],
    'product': ['Laptop', 'Phone', 'Tablet'],
    'amount': [1500, 800, 1200]
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('sales_data.csv', index=False)

print("CSV file created successfully.")
