import json


def remove_duplicate_words(data):
    """
    This function removes duplicate words between categories in a JSON structure,
    keeping the entry with the higher ratio.

    Args:
        data: A dictionary containing "Severe Lexicons" and "NonSevere Lexicons" keys.

    Returns:
        A modified dictionary with duplicates removed.
    """
    combined_lexicons = {**data["Severe Lexicons"], **data["NonSevere Lexicon"]}
    for word, word_data in combined_lexicons.items():
        if word in data["Severe Lexicons"] and word in data["NonSevere Lexicon"]:
            # Keep the entry with the higher ratio
            higher_ratio_category = "Severe Lexicons" if data["Severe Lexicons"][word]["ratio"] > data["NonSevere Lexicon"][word]["ratio"] else "NonSevere Lexicon"
            lower_ratio_category = "NonSevere Lexicon" if higher_ratio_category == "Severe Lexicons" else "Severe Lexicons"
            del data[lower_ratio_category][word]

    return data


def main():
    # Replace 'lexicon.json' with your actual filename
    with open('merged_lexicon_Exp1_Exp2.json', 'r') as f:
        lexicon = json.load(f)

    filtered_lexicon_exp5 = remove_duplicate_words(lexicon)
    print(filtered_lexicon_exp5)

    # Write filtered lexicon to a new JSON file (replace with desired filename)
    with open('merged_lexicon_Exp1_Exp2_Filtered.json', 'w') as f:
        json.dump(filtered_lexicon_exp5, f, indent=4)

if __name__ == "__main__":
    main()

# # Example usage
# data = {
#     "Severe Lexicons": {
#         "recursion": {"ratio": 1.0},
#         "tvt": {"ratio": 0.0},
#         "productivity": {"ratio": 0.4},
#     },
#     "NonSevere Lexicons": {
#         "logins": {"ratio": 1.0},
#         "recursion": {"ratio": 0.8},
#         "extendable": {"ratio": 1.0},
#     }
# }

# modified_data = remove_duplicate_words(data.copy())
# print(modified_data)






    
    
    
    
    
    
    
    
    
   