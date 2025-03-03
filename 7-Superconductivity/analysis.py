import pandas as pd

# Read the ODS file
df = pd.read_excel("data.ods", engine="odf", sheet_name=None)  # Use sheet_name=None to read all sheets

# Print the first few rows of the first sheet
for sheet, data in df.items():
    print(f"Sheet: {sheet}")
    print(data.head())  # Display first few rows
