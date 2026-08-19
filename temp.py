import pandas as pd

# Load the Parquet file into memory
df = pd.read_parquet('C:\\Users\\kavan\\OneDrive\\Desktop\\IIITH\\sem3\\IRE\\Assignment1\\IRE-Assignment-1\\data\\raw\\ebnerd\\demo\\articles.parquet', engine='pyarrow')

# Export the data to a CSV file (index=False prevents writing row numbers)
df.to_csv('output_file.csv', index=False)