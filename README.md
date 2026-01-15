# Review Sentiment Analysis
This project implements a sentiment analysis system for customer reviews for an electronic store.

A RoBERTa transformer model is fine-tuned on Amazon reviews and used to classify reviews into negative, neutral or positive sentiments.

## Setup instructions
1. Create and activate a virtual environment on the terminal (for Windows)

   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   
2. Install dependencies

   ```
   pip install -r requirements.txt
   ```
   
3. Install GPU Support for faster training (Optional, if you have NVIDIA)

   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

1. Run the training script, the model will be saved locally to ".\sentiment_model"

   ```
   python train.py
   ```
  
2. Create input data CSV file

   ```
   python create_input_file.py
   ```
   after running this command, you should see a file called "input_reviews.csv" saved in the directory.
   
3. Run sentiment analysis

   ```
   python main.py
   ```
