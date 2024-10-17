# Merge Json files for Eclipse and Firefox lexicon from Exp 1 and 2 
import json


import json


def merge_json_lexicons(file1, file2):
    """Merges lexicon files, combining severe and non-severe categories while removing duplicates within each category.

    Args:
        file1 (str): Path to the first JSON lexicon file.
        file2 (str): Path to the second JSON lexicon file.

    Returns:
        dict: Merged lexicon with categories and unique words.
    """

    with open(file1, 'r') as f:
        data1 = json.load(f)

    with open(file2, 'r') as f:
        data2 = json.load(f)

    merged_lexicon = {}
    # Iterate over categories in both dictionaries
    for category in set(data1.keys() | data2.keys()):
        merged_lexicon[category] = {}
        # Track words seen in this category
        seen_words = set()
        for word, info in {**data1.get(category, {}), **data2.get(category, {})}.items():
            if word not in seen_words:
                merged_lexicon[category][word] = info
                seen_words.add(word)
            else:
                # Check ratio and keep the one with higher ratio
                if info['ratio'] > merged_lexicon[category][word]['ratio']:
                    merged_lexicon[category][word] = info
                # Print duplicates for analysis (remove this line if not needed)
                print(f"Duplicate found in category {category}: {word} (keeping higher ratio)")

    return merged_lexicon


# Example usage
merged_lexicon = merge_json_lexicons('static_dictionary_Firefox_Lsvm11_bestc.json', 'static_dictionary_eclipse_LinearSVM_bestc.json')

# Write merged lexicon to a new file (optional)
with open('merged_lexicon_Exp10_Exp11.json', 'w') as f:
    json.dump(merged_lexicon, f, indent=2)


    
# import json

# def merge_json_lexicons(file1, file2):
#     """
#     Merges lexicon files, combining severe and non-severe categories
#     while removing duplicates within each category.

#     Args:
#       file1 (str): Path to the first JSON lexicon file.
#       file2 (str): Path to the second JSON lexicon file.

#     Returns:
#       dict: Merged lexicon with categories and unique words.
#     """
#     with open(file1, 'r') as f:
#         data1 = json.load(f)
#     with open(file2, 'r') as f:
#         data2 = json.load(f)

#     merged_lexicon = {}
#     for category in ["Severe Lexicons", "NonSevere Lexicon"]:
#         merged_lexicon[category] = list(set(data1[category] + data2[category]))

#     return merged_lexicon

# # Example usage
# merged_lexicon = merge_json_lexicons("static_dictionary_eclipse.json", "static_dictionary_Firefox.json")

# # Write merged lexicon to a new file 
# with open("merged_lexicon_Exp1_Exp2.json", "w") as f:
#     json.dump(merged_lexicon, f,indent=2)
   


