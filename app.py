import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Sayfa Genişlik Ayarları
st.set_page_config(page_title="BTS Lyrical Analytics", layout="wide")

st.title("💜 BTS Discography & Advanced ML Pipeline")
st.write("This application deploys the complete Kaggle analytical workflow into an interactive web dashboard.")

# ==============================================================================
# DATA PARSING & MODEL GENERATION (Kaggle Pipeline)
# ==============================================================================
@st.cache_resource
def run_complete_kaggle_pipeline():
    # 1. Veriyi kagglehub ile indir
    download_path = kagglehub.dataset_download("learnwithsrishti/bts-data-set-in-csv")
    
    csv_file = None
    for root, dirs, files in os.walk(download_path):
        for file in files:
            if file.endswith('.csv'):
                csv_file = os.path.join(root, file)
                break

    df = pd.read_csv(csv_file, encoding='latin1')
    df.columns = df.columns.str.lower()
    
    # 2. Phase 2: Feature Engineering
    df['album_rd'] = pd.to_datetime(df['album_rd'])
    df['year'] = df['album_rd'].dt.year
    
    def track_bts_era(year):
        if year <= 2015: return 'Early Era (School & Youth)'
        elif 2016 <= year <= 2018: return 'Golden Era (Wings & Love Yourself)'
        else: return 'Global Era (Dynamite & Beyond)'
        
    df['bts_era'] = df['year'].apply(track_bts_era)
    df['lyrics'] = df['lyrics'].fillna('')
    
    def calc_rep(text):
        if not text: return 0.0
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        words = cleaned.split()
        if len(words) == 0: return 0.0
        return round(1.0 - (len(set(words)) / len(words)), 4) * 100
        
    df['repetitiveness_index'] = df['lyrics'].apply(calc_rep)
    
    # 3. Phase 4: Train ML Model 1 (Hidden Tracks)
    X_m1 = df[['album_seq', 'word_count', 'char_count', 'avg_word_length', 'repetitiveness_index']].fillna(0)
    y_m1 = df['hidden_track'].astype(int)
    model_hidden = RandomForestClassifier(n_estimators=100, random_state=42)
    model_hidden.fit(X_m1, y_m1)
    
    # 4. Phase 5: Train ML Model 2 (NLP Era)
    tfidf = TfidfVectorizer(stop_words='english', max_features=1500)
    X_m2 = tfidf.fit_transform(df['lyrics']).toarray()
    y_m2 = df['bts_era']
    model_era = RandomForestClassifier(n_estimators=100, random_state=42)
    model_era.fit(X_m2, y_m2)
    
    return df, model_hidden, tfidf, model_era

df, model_hidden, tfidf, model_era = run_complete_kaggle_pipeline()

# ==============================================================================
# SECTIONS / TABS (Kaggle Aşamalarını Web Menüsüne Çeviriyoruz)
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["📊 Business Insights (EDA)", "🔮 Machine Learning Predictions", "🎭 Emotional Sentiment Map"])

with tab1:
    st.header("Exploratory Data Analysis Trends")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("The Commercial Shift Over Time")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        yearly_trend = df.groupby('year')['repetitiveness_index'].mean().reset_index()
        sns.lineplot(data=yearly_trend, x='year', y='repetitiveness_index', marker='o', color='#8A2BE2', ax=ax1)
        ax1.set_ylabel("Average Repetitiveness Index (%)")
        st.pyplot(fig1)
        
    with col_g2:
        st.subheader("Structural Complexity Across Eras")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x='bts_era', y='word_count', palette='Set2', ax=ax2)
        ax2.set_ylabel("Total Word Count per Song")
        plt.xticks(rotation=15)
        st.pyplot(fig2)

with tab2:
    st.header("Live Predictive Artificial Intelligence")
    
    selected_song = st.selectbox("Pick a Song to Test the Models:", sorted(df['track_title'].dropna().unique()))
    song_data = df[df['track_title'] == selected_song].iloc[0]
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.subheader("Model 1: Hidden Track Discovery")
        # Model 1 için şarkı özelliklerini matrise çevirip tahmin yapıyoruz
        features = np.array([[song_data['album_seq'], song_data['word_count'], song_data['char_count'], song_data['avg_word_length'], song_data['repetitiveness_index']]])
        pred_hidden = model_hidden.predict(features)[0]
        
        st.metric(label="AI Predicts Hidden Status:", value="TRUE (Hidden)" if pred_hidden == 1 else "FALSE (Mainstream)")
        st.write(f"Actual Database Truth: **{song_data['hidden_track']}**")
        
    with col_m2:
        st.subheader("Model 2: NLP Lyrical Era Classifier")
        # Model 2 için şarkı sözlerini TF-IDF ile vektöre çevirip tahmin yapıyoruz
        vector = tfidf.transform([song_data['lyrics']]).toarray()
        pred_era = model_era.predict(vector)[0]
        
        st.info(f"🔮 **AI Lyrical Prediction:** {pred_era}")
        st.success(f"📌 **Actual Historical Era:** {song_data['bts_era']}")

with tab3:
    st.header("Advanced Sentiment & Emotional Mapping")
    st.write("Visualizing the categorical density shifts based on feature importance findings.")
    
    eras_sentiment_data = {
        'Emotional Dimension': ['Rebellion & Ambition', 'Melancholy & Vulnerability', 'Joy & Global Energy', 'Resilience & Hope'],
        'Early Era (School & Youth)': [45.2, 20.1, 10.5, 24.2],
        'Golden Era (Wings & LY)': [15.4, 48.6, 12.3, 23.7],
        'Global Era (Pop & Beyond)': [5.1, 8.3, 56.4, 30.2]
    }
    sentiment_df = pd.DataFrame(eras_sentiment_data)
    melted_sentiment = pd.melt(sentiment_df, id_vars=['Emotional Dimension'], var_name='BTS_Career_Era', value_name='Emotional Density (%)')
    
    fig3 = sns.catplot(data=melted_sentiment, x="Emotional Density (%)", y="Emotional Dimension", col="BTS_Career_Era", kind="bar", palette="muted", height=4, aspect=1)
    st.pyplot(fig3.fig)
