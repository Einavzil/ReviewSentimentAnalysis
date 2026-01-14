""" 
This script creates an input file from the dataset used in training,
skipping the first 15k samples to ensure no data that was in training will be used.
"""

import pandas as pd
from datasets import load_dataset

def create_input_file():
    """
    Creates an input CSV file with 1000 reviews per class, formatting the text.
    """
    # connect to dataset
    data_url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Electronics.jsonl"
    dataset_stream = load_dataset("json", data_files=data_url, split="train", streaming=True)

    # skip the first 15k samples
    dataset_stream = dataset_stream.skip(15000)

    # collect reviews into a list, with 5 reviews per class
    reviews = []
    target_per_class = 1000
    class_counts = {1:0, 3:0, 5:0}

    for row in dataset_stream:
        rating = int(row['rating'])

        if rating in class_counts and class_counts[rating] < target_per_class:
            formatted_text = row['text'].replace('\n', ' ').strip()

            reviews.append({'text': formatted_text})
            class_counts[rating] += 1

        if all(count >= target_per_class for count in class_counts.values()):
            break

    # create dataframe and save to csv
    df = pd.DataFrame(reviews)
    output_file = "input_reviews.csv"
    df.to_csv(output_file, index=False)
    print(f"Input file created at {output_file} with {len(df)} reviews.")

if __name__ == "__main__":
    create_input_file()
