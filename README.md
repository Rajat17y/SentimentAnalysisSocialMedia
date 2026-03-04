# Sentiment Analysis on Social Media

An AI-powered web application that scrapes YouTube comments in real-time and performs deep learning-based sentiment analysis to visualize the "vibe" of any video’s comment section.

## Features

1. Live Scraping: Connects to the YouTube Data API v3 to fetch up to 1,000 comments per video.
2. Deep Learning Engine: Uses a TensorFlow Keras model to classify sentiments into **Positive**, **Negative**, or **Neutral**.
3. Interactive Visualizations: Donut Chart: Shows the percentage distribution of sentiments.
4. Detailed Analytics: Searchable data table with confidence scores.
5. Expandable comment view for granular reading.
6. Robust Preprocessing: Automated text cleaning including regex-based noise reduction and NLTK stop-word removal.

---

## Tech Stack

 - Frontend: [Streamlit](https://streamlit.io/)
 - Machine Learning: TensorFlow/Keras, Scikit-learn (Joblib)
 - Data Handling: Pandas, NumPy
 - Visualization: Plotly Express & Graph Objects
 - API: Google API Client (YouTube Data API v3)
 - NLP: NLTK (Natural Language Toolkit)


## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Rajat17y/SentimentAnalysisSocialMedia.git
```


2. **Install dependencies:**
```bash
pip install -r requirements.txt
```


3. **Set your API Key:**
Open `userinterRUN.py` and paste your YouTube API key:
```python
api_key = 'YOUR_API_KEY_HERE'

```



---

## Usage

Run the application using Streamlit:

```bash
streamlit run userinterRUN.py

```

1. Paste a **YouTube Video URL** (e.g., `https://www.youtube.com/watch?v=...`) or a **Video ID** into the input box.
2. Click **Analyze Comments**.
3. View the generated charts and the detailed sentiment breakdown.

---

## Model Logic

1. **Preprocessing:** Text is converted to lowercase, special characters are removed, and NLTK stopwords (excluding "not") are filtered out.
2. **Vectorization:** Text is converted to sequences and padded to a `MAX_LEN` of 100.
3. **Inference:** The model predicts the probability of each class.
4. **Metrics:** * **Positivity Index:** Calculated as `(Positive Comments / (Positive + Negative Comments)) * 100`.

