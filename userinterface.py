import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from googleapiclient.discovery import build
import plotly.express as px
import plotly.graph_objects as go

api_key = ''

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🧠",
    layout="wide"
)

# --- CONSTANTS ---
MAX_LEN = 100 

# --- LOAD ASSETS (Cached for Speed) ---
@st.cache_resource
def load_assets():
    try:
        # Download NLTK resources silently
        nltk.download('stopwords', quiet=True)
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        stop_words_set = set(all_stopwords)

        # Load Model & Helpers
        model = tf.keras.models.load_model('modelA.keras')
        tokenizer = joblib.load('tokenizerA.joblib')
        label_encoder = joblib.load('encoderA.joblib')
        
        return model, tokenizer, label_encoder, stop_words_set
    except Exception as e:
        return None, None, None, None

# Load them immediately
model, tokenizer, label_encoder, stop_words_set = load_assets()

# --- HELPER FUNCTIONS ---

def clean_text(text):
    text = str(text)
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in stop_words_set]
    return ' '.join(review)

def predict_batch(texts):
    # 1. Clean
    cleaned_texts = [clean_text(t) for t in texts]
    
    # 2. Tokenize
    seqs = tokenizer.texts_to_sequences(cleaned_texts)
    
    # 3. Pad
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 4. Predict
    pred_probs = model.predict(padded, verbose=0)
    
    # 5. Decode
    pred_indices = np.argmax(pred_probs, axis=1)
    pred_labels = [label_encoder.classes_[i] for i in pred_indices]
    confidences = [pred_probs[i][idx] for i, idx in enumerate(pred_indices)]
    
    return pred_labels, confidences

def get_youtube_comments(api_key, video_id, max_count=50):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    # Get Video Title first
    vid_request = youtube.videos().list(part="snippet", id=video_id)
    vid_response = vid_request.execute()
    if not vid_response['items']:
        return None, []
    video_title = vid_response['items'][0]['snippet']['title']

    # Get Comments
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_count,
        textFormat="plainText"
    )
    response = request.execute()

    comments = []
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
        comments.append({"Author": author, "Comment": comment})
        
    return video_title, comments

def extract_video_id(url):
    # Regex to handle various YouTube URL formats
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return url # Return original if no match (assume it is the ID)

# --- GUI LAYOUT ---

# Sidebar for Inputs
with st.sidebar:
    st.markdown("### Model Status")
    if model:
        st.success("✅ Model Loaded")
    else:
        st.error("❌ Model Missing. Check .keras files.")

# Main Page
st.title("📺 YouTube Sentiment Dashboard")
st.markdown("Analyze the emotions of any YouTube video comment section using Deep Learning.")

# Input Section
col1, col2 = st.columns([3, 1])
with col1:
    video_url = st.text_input("YouTube Video URL or ID", placeholder="e.g., https://www.youtube.com/watch?v=Wl959QnD3lM")
with col2:
    st.write("") # Spacer
    st.write("") # Spacer
    analyze_btn = st.button("🚀 Analyze Comments", type="primary", use_container_width=True)

if analyze_btn:
    if not video_url:
        st.warning("⚠️ Please enter a Video URL.")
    else:
        video_id = extract_video_id(video_url)
        
        with st.spinner(f"Fetching comments for Video ID: {video_id}..."):
            try:
                # 1. Fetch Data
                video_title, raw_comments = get_youtube_comments(api_key, video_id, max_count=100)
                
                if not raw_comments:
                    st.error("No comments found or Invalid Video ID.")
                else:
                    st.subheader(f"🎬 {video_title}")
                    
                    # 2. Predict
                    df = pd.DataFrame(raw_comments)
                    labels, scores = predict_batch(df['Comment'].tolist())
                    
                    df['Sentiment'] = labels
                    df['Confidence'] = scores
                    
                    # 3. Metrics
                    st.markdown("---")
                    total = len(df)
                    pos_count = len(df[df['Sentiment'] == 'Positive'])
                    neg_count = len(df[df['Sentiment'] == 'Negative'])
                    neu_count = len(df[df['Sentiment'] == 'Neutral'])
                    
                    # Calculate a "Positivity Score" (Simple ratio)
                    # Avoid division by zero
                    total_emotional = pos_count + neg_count
                    if total_emotional > 0:
                        positivity_ratio = (pos_count / total_emotional) * 100
                    else:
                        positivity_ratio = 0
                        
                    # Display Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Comments", total)
                    m2.metric("Positive 🟢", pos_count)
                    m3.metric("Negative 🔴", neg_count)
                    m4.metric("Positivity Index", f"{positivity_ratio:.1f}%")

                    # 4. Charts
                    row1_col1, row1_col2 = st.columns([1, 1])
                    
                    with row1_col1:
                        st.subheader("Sentiment Distribution")
                        # Custom colors for sentiments
                        color_map = {
                            "Positive": "#00CC96", # Green
                            "Negative": "#EF553B", # Red
                            "Neutral":  "#636EFA", # Blue
                            "Irrelevant": "#FECB52" # Yellow
                        }
                        
                        fig_pie = px.pie(
                            df, 
                            names='Sentiment', 
                            hole=0.4, # Makes it a donut chart
                            color='Sentiment',
                            color_discrete_map=color_map
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with row1_col2:
                        st.subheader("Overall 'Vibe' Gauge")
                        
                        # Gauge Chart
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = positivity_ratio,
                            title = {'text': "Positivity Score (0-100)"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "lightgray"},
                                'steps': [
                                    {'range': [0, 40], 'color': "#EF553B"},
                                    {'range': [40, 60], 'color': "#FECB52"},
                                    {'range': [60, 100], 'color': "#00CC96"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': positivity_ratio
                                }
                            }
                        ))
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    # 5. Data Table
                    st.subheader("📝 Comment Analysis")
                    
                    # Add color formatting to the dataframe display
                    def highlight_sentiment(val):
                        if val == 'Positive': return 'background-color: #d4edda; color: green'
                        elif val == 'Negative': return 'background-color: #f8d7da; color: red'
                        return ''

                    st.dataframe(
                        df[['Author', 'Sentiment', 'Confidence', 'Comment']],
                        use_container_width=True,
                        hide_index=True
                    )
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")