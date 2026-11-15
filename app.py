import streamlit as st
import pandas as pd
import numpy as np
from src.data_manager import DataManager
from src.emotion_detector import EmotionDetector
from src.recommenders import CollaborativeRecommender, HybridRecommender

st.set_page_config(page_title="Mood-Based Recommender", page_icon="🎶", layout="centered")

@st.cache_resource
def load_resources():
    """Load Spotify data and the public GoEmotions model from Hugging Face."""
    # This is a real, already fine-tuned GoEmotions model. Do not use the
    # non-existent local path './fine-tuned-goemotions-model'.
    MODEL_ID = "SamLowe/roberta-base-go_emotions"
    SPOTIFY_DATA_PATH = "data/data.csv"

    data_manager = DataManager(SPOTIFY_DATA_PATH)
    emotion_detector = EmotionDetector(MODEL_ID, MODEL_ID)

    # The original project creates synthetic ratings (not real user history).
    # Keep this modest for Streamlit Cloud memory/startup limits.
    songs = data_manager.get_all_data()["song_id"].to_numpy()
    rng = np.random.default_rng(42)
    ratings_df = pd.DataFrame({
        "userID": rng.integers(0, 500, 50_000),
        "songID": rng.choice(songs, 50_000),
        "rating": rng.integers(1, 6, 50_000),
    })

    collab_recommender = CollaborativeRecommender(
        ratings_df, data_manager.get_all_data()
    )
    return HybridRecommender(
        emotion_detector=emotion_detector,
        collab_recommender=collab_recommender,
        data_manager=data_manager,
    )

try:
    with st.spinner("Loading Spotify data and emotion model (first launch may take a few minutes)..."):
        hybrid_recommender = load_resources()
except Exception as exc:
    st.error("The app could not load its data/model. Check the Streamlit Cloud logs for details.")
    st.exception(exc)
    st.stop()

st.title("🎶 Hybrid Mood-Based Recommender")
st.write("Enter how you're feeling, select a user ID, and get personalized song recommendations.")
user_text = st.text_input("How are you feeling right now?", "I am feeling happy and excited today!")
sample_user_id = st.selectbox("Select a User ID (for personalization)", (42, 101, 256, 312, 499))
context_option = st.selectbox("Select a Context (Optional)", ("None", "Evening", "Workout"))

if st.button("Get Recommendations"):
    if not user_text.strip():
        st.error("Please enter some text to describe your mood.")
    else:
        detected_emotions = hybrid_recommender.emotion_detector.predict(user_text)
        st.info(f"🧠 **Detected Mood(s):** {', '.join(detected_emotions)}")
        context = None
        if context_option == "Evening":
            context = {"time_of_day": "evening"}
        elif context_option == "Workout":
            context = {"activity": "workout"}
        with st.spinner("Finding songs that match your mood..."):
            recommendations = hybrid_recommender.recommend(
                user_text=user_text, user_id=sample_user_id, context=context
            )
        st.success("Here are your personalized recommendations!")
        st.dataframe(
            recommendations[["name", "artists", "valence", "energy", "mood_match_score"]],
            use_container_width=True,
        )
