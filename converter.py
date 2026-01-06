import pandas as pd
import gzip
import json

# 🔧 CONFIGURE: Update this to your file path
input_file = 'Electronics_5.json.gz'   # Your downloaded file
output_file = 'amazon_reviews.csv'   # Name of the CSV you want to create
num_lines = 10000                    # How many reviews to extract (adjust as needed)

print("Starting conversion...")

# List to store data
data = []

# Open .json.gz file and read line by line
with gzip.open(input_file, 'rt', encoding='utf8') as f:
    for i, line in enumerate(f):
        if i >= num_lines:
            break  # Stop after N lines
        try:
            # Parse each line as JSON
            row = json.loads(line.strip())
            data.append(row)
        except:
            continue  # Skip corrupted/bad lines

# Convert list of dicts to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv(output_file, index=False)

print(f"✅ Done! Saved {len(df)} rows to '{output_file}'")