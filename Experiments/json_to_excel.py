import pandas as pd
import json

# Specify the path to your JSON file
json_file_path = "static_dictionary_Firefox_Lsvm11.json"

# Read the JSON data from the file
with open(json_file_path) as json_file:
    data = json.load(json_file)

# Extract terms and ratios from each category
severe_terms = list(data["Severe Lexicons"].keys())
severe_ratios = [data["Severe Lexicons"][term]["ratio"] for term in severe_terms]

nonsevere_terms = list(data["NonSevere Lexicon"].keys())
nonsevere_ratios = [data["NonSevere Lexicon"][term]["ratio"] for term in nonsevere_terms]

# Create dataframes for severe and nonsevere lexicons
severe_df = pd.DataFrame({"Term": severe_terms, "Ratio": severe_ratios})
nonsevere_df = pd.DataFrame({"Term": nonsevere_terms, "Ratio": nonsevere_ratios})

# Create Excel writer
with pd.ExcelWriter("static_dictionary_Firefox_Lsvm11.xlsx") as writer:
    # Write dataframes to separate sheets
    severe_df.to_excel(writer, sheet_name="Severe Lexicons", index=False)
    nonsevere_df.to_excel(writer, sheet_name="NonSevere Lexicons", index=False)

print("Excel file 'lexicons.xlsx' created successfully!")







