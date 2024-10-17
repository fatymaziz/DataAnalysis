import pandas as pd
import json

# Load the data
data = pd.read_json("lexicon_results15_vader.json")

# Calculate and update F1Score_mean
for index, entry in data.iterrows():
    f1_scores = entry["F1Score"]
    f1_mean = sum(f1_scores) / len(f1_scores)
    data.at[index, "F1Score_mean"] = f1_mean

# Print updated data
print(data.to_json(orient='records', indent=4))

# Save the updated data
data.to_json('lexicon_results15_vader_updated.json', orient='records', indent=4)
