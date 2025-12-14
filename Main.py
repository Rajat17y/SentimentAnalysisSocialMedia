import os
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from googleapiclient.discovery import build

# --- CONFIGURATION ---
# 1. Paste your Google API Key here
API_KEY = '' 

# 2. Paste the Video ID you want to analyze
# (Found in the URL: youtube.com/watch?v=VIDEO_ID)
VIDEO_ID = 'DSZGVqC42XI'  # Example: "Me at the zoo"

# 3. Model Constants (MUST match your training!)
MAX_LEN = 100

# --- 1. LOAD MODEL & ASSETS ---
print("Loading model and assets...")
try:
    model = tf.keras.models.load_model('modelA.keras')
    tokenizer = joblib.load('tokenizerA.joblib')
    label_encoder = joblib.load('encoderA.joblib')
    
    # Load stopwords
    nltk.download('stopwords')
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    stop_words_set = set(all_stopwords)
    print("Assets loaded successfully.\n")
except Exception as e:
    print(f"Error loading assets: {e}")
    print("Ensure your .keras and .joblib files are in this folder.")
    exit()

# --- 2. DEFINE PREDICTION FUNCTION ---
def predict_sentiment(text):
    # Clean
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in stop_words_set]
    cleaned_text = ' '.join(review)
    
    # Tokenize & Pad
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Predict
    pred_probs = model.predict(padded, verbose=0) # verbose=0 hides the progress bar
    pred_index = np.argmax(pred_probs)
    sentiment = label_encoder.classes_[pred_index]
    confidence = pred_probs[0][pred_index]
    
    return sentiment, confidence

# --- 3. FETCH YOUTUBE COMMENTS ---
def get_video_comments(api_key, video_id, max_results=20):
    print(f"Fetching comments for video ID: {video_id}...")
    
    # Build the YouTube client
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Request comments
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    response = request.execute()

    comments = []
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
        comments.append((author, comment))
    
    return comments

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # Get comments from YouTube
        comments_data = get_video_comments(API_KEY, VIDEO_ID, max_results=20)
        
        print(f"\n{'='*20} RESULTS {'='*20}\n")
        
        for author, text in comments_data:
            # Predict Sentiment
            sentiment, conf = predict_sentiment(text)
            
            # Print Result
            print(f"User: {author}")
            print(f"Comment: \"{text}\"")
            print(f"Prediction: {sentiment} ({conf*100:.1f}%)")
            print("-" * 40)
            
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your API Key and Video ID.")