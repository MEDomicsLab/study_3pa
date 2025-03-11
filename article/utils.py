"""

"""
import os
import json


def extract_results_data(folder_path):
    data_list = []
    print(f"{os.getcwd()=}")

    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".MED3paResults"):  # Check if the file is a MED3paResults file
            file_path = os.path.join(folder_path, file_name)

            # Open and read the JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)  # Load JSON content
                    data['filename'] = file_name
                    data_list.append(data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {file_name}")

    return data_list

if __name__ == "__main__":
    data = extract_results_data("results")
