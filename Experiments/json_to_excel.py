import pandas as pd
import json

# Specify the path to your JSON file
json_file_path = "Worlist_frequentword_eclipse_THR_AFTER_negationhandled2.json"

# Read the JSON data from the file
with open(json_file_path) as json_file:
    data = json.load(json_file)

# Convert JSON data to DataFrames
severe_df = pd.DataFrame(data["Severe"])
nonsevere_df = pd.DataFrame(data["NonSevere"])

# Create a Pandas Excel writer using xlsxwriter as the engine
with pd.ExcelWriter("Worlist_frequentword_eclipse_THR_AFTER_negation.xlsx", engine="xlsxwriter") as writer:
    # Write DataFrames to separate sheets
    severe_df.to_excel(writer, sheet_name="Severe", index=False)
    nonsevere_df.to_excel(writer, sheet_name="NonSevere", index=False)

print("Excel file 'Worlist_frequentword_eclipse_THR_After_negation.xlsx' created successfully!")








# import pandas as pd
# import json

# # Specify the path to your JSON file
# json_file_path = "Worlist_frequentword_eclipse_THR_BEFORE_negation.json"

# # Read the JSON data from the file
# with open(json_file_path) as json_file:
#     data = json.load(json_file)

# # Extract terms and ratios from each category
# severe_terms = list(data["Severe"].keys())
# severe_ratios = [data["Severe"][term]["ratio"] for term in severe_terms]

# nonsevere_terms = list(data["NonSevere"].keys())
# nonsevere_ratios = [data["NonSevere"][term]["ratio"] for term in nonsevere_terms]

# # Create dataframes for severe and nonsevere lexicons
# severe_df = pd.DataFrame({"Term": severe_terms, "Ratio": severe_ratios})
# nonsevere_df = pd.DataFrame({"Term": nonsevere_terms, "Ratio": nonsevere_ratios})

# # Create Excel writer
# with pd.ExcelWriter("Worlist_frequentword_eclipse_THR_BEFORE_negation.xlsx") as writer:
#     # Write dataframes to separate sheets
#     severe_df.to_excel(writer, sheet_name="Severe", index=False)
#     nonsevere_df.to_excel(writer, sheet_name="NonSevere", index=False)

# print("Excel file 'lexicons.xlsx' created successfully!")







