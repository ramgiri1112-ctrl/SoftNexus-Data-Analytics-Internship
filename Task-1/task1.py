import pandas as pd
# Load data   df = pd.read_csv("messy_sales_data.csv")
# Remove duplicates & irrelevant columns
df = df.drop_duplicates().drop(columns=["notes", "temp_id"])
# Handle missing values
df["revenue"].fillna(df["revenue"].median(), inplace=True) 
df.dropna(subset=["customer_id"], inplace=True)
# Convert data types
df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce") 
df["price"] = pd.to_numeric(df["price"].str.replace("$", ""))
# Standardize formats
df["product"] = df["product"].str.lower().str.strip() 
df["region"] = df["region"].replace({"west": "Western", "SOUTH": "Southern"})  
