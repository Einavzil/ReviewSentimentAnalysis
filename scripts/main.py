"""
main.py is the entry point for sentiment analysis on product reviews.
Reads reviews from an input CVS file, loads a pre-trained sentiment model,
and writes the sentiment predictions to an output CSV file.
"""

import pandas as pd
import os
from sentiment import SentimentModel
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

def main():
    # Setup Files
    input_file = "input_reviews.csv"
    output_file = "output_reviews.csv"

    # valiadate input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    # initalize sentiment model
    try:
        analyzer = SentimentModel()
    except Exception as err:
        print(f"Error initializing SentimentModel: {err}")
        return
    
    # read input reviews
    try:
        df = pd.read_csv(input_file)
    except Exception as err:
        print(f"Error reading input file: {err}")
        return
    
    if "text" not in df.columns:
        print("Input file must contain a 'text' column.")
        return
    
    # analyze sentiment for each review
    results = []
    confidences = []

    for index, row in df.iterrows():
        review = str(row['text'])

        # handle empty or NaN reviews
        if review.lower() == 'nan' or not review.strip():
            results.append("neutral")
            confidences.append(0.0)
            continue

        try:
            label, score = analyzer.predict(review)
            results.append(label)
            confidences.append(score)
        except Exception as err:
            print(f"Error processing review at index {index}: {err}")
            results.append("error")
            confidences.append(0.0)

    # save results to output file
    df['sentiment'] = results
    df['confidence'] = confidences
    
    df.to_csv(output_file, index=False)
    print(f"Success! Saved results to {output_file}.")

    # print summary result to console, interate through all unique labels
    print("\nSentiment Counts:")
    for label, count in df['sentiment'].value_counts().items():
        print(f"{label}: {count}")

if __name__ == "__main__":
    main()