import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI Music Hit Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# LOAD MODELS
# ==========================================
MODEL_DIR = "models/"

@st.cache_resource
def load_ml_assets():
    try:
        model = tf.keras.models.load_model(MODEL_DIR + "hit_dnn_model.h5")
        scaler = joblib.load(MODEL_DIR + "scaler.pkl")
        features_list = joblib.load(MODEL_DIR + "features.pkl")
        # In the latest script, we use best_thresh.pkl
        best_thresh = joblib.load(MODEL_DIR + "best_thresh.pkl")
        
        # Load the new metrics files for the dashboard
        df_metrics = joblib.load(MODEL_DIR + "metrics_df.pkl")
        roc_data = joblib.load(MODEL_DIR + "roc_data.pkl")
        feature_ranges = joblib.load(MODEL_DIR + "feature_ranges.pkl")
        
        return model, scaler, features_list, best_thresh, df_metrics, roc_data, feature_ranges, None
    except Exception as e:
        return None, None, None, None, None, None, None, str(e)

model, scaler, feature_names, best_thresh, df_metrics, roc_data, feature_ranges, load_error = load_ml_assets()

# ==========================================
# UI HEADER
# ==========================================
st.title("🎵 AI Music Hit Predictor")
st.markdown("##### Powered by Deep Neural Networks")

if model is None:
    st.error(f"⚠️ **Model files not found!**\nError Detail: `{load_error}`")
    st.stop()

# ==========================================
# INPUT FORM
# ==========================================
st.markdown("### 🎛️ Tune Your Track")
st.write("")

# 3-Column Wide Dashboard Layout
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("#### 🎵 Vibe & Energy")
    st.write("")
    
    # Helper to get range safely (with crash-prevention)
    def get_range(feat, key, default):
        if feature_ranges and feat in feature_ranges:
            f_min = float(feature_ranges[feat]["min"])
            f_max = float(feature_ranges[feat]["max"])
            f_mean = float(feature_ranges[feat]["mean"])
            
            # Streamlit sliders crash if min == max. 
            # If the dataset has no variance for a feature (e.g. sentiment score in FAST_MODE)
            # we force a microscopic range so the UI still renders.
            if f_min == f_max:
                f_max = f_min + 0.01 
            
            if key == "min": return f_min
            if key == "max": return f_max
            if key == "mean": return max(f_min, min(f_max, f_mean))
        return default

    danceability = st.slider("💃 Danceability", 
                             get_range("danceability", "min", 0.04), 
                             get_range("danceability", "max", 0.99), 
                             get_range("danceability", "mean", 0.53), 0.01)
    
    energy = st.slider("⚡ Energy", 
                       get_range("energy", "min", 0.0), 
                       get_range("energy", "max", 1.0), 
                       get_range("energy", "mean", 0.67), 0.01)
    
    valence = st.slider("😊 Valence (Happy/Sad)", 
                        get_range("valence", "min", 0.0), 
                        get_range("valence", "max", 1.0), 
                        get_range("valence", "mean", 0.46), 0.01)
    
    tempo = st.number_input("🥁 Tempo (BPM)", 
                            get_range("tempo", "min", 30.0), 
                            get_range("tempo", "max", 250.0), 
                            get_range("tempo", "mean", 122.8))

with col2:
    st.markdown("#### 🎧 Acoustics & Mix")
    st.write("")
    loudness = st.slider("🔊 Loudness (dB)", 
                         get_range("loudness", "min", -45.0), 
                         get_range("loudness", "max", 0.0), 
                         get_range("loudness", "mean", -7.9), 0.1)
    
    acousticness = st.slider("🎸 Acousticness", 
                             get_range("acousticness", "min", 0.0), 
                             get_range("acousticness", "max", 1.0), 
                             get_range("acousticness", "mean", 0.24), 0.01)
    
    instrumentalness = st.slider("🎷 Instrumentalness", 
                                 get_range("instrumentalness", "min", 0.0), 
                                 get_range("instrumentalness", "max", 1.0), 
                                 get_range("instrumentalness", "mean", 0.11), 0.01)
    
    liveness = st.slider("🏟️ Liveness", 
                         get_range("liveness", "min", 0.0), 
                         get_range("liveness", "max", 1.0), 
                         get_range("liveness", "mean", 0.22), 0.01)

with col3:
    st.markdown("#### 📈 Metadata & Reach")
    st.write("")
    speechiness = st.slider("🗣️ Speechiness", 
                            get_range("speechiness", "min", 0.02), 
                            get_range("speechiness", "max", 0.97), 
                            get_range("speechiness", "mean", 0.09), 0.01)
    
    sentiment_score = st.slider("📝 Lyrics Sentiment (-1 to 1)", 
                                get_range("sentiment_score", "min", -1.0), 
                                get_range("sentiment_score", "max", 1.0), 
                                get_range("sentiment_score", "mean", 0.0), 0.01)
    
    followers = st.number_input("👥 Artist Followers", 
                                int(get_range("total_artist_followers", "min", 0)), 
                                int(get_range("total_artist_followers", "max", 300000000)), 
                                int(get_range("total_artist_followers", "mean", 2500000)))
    
    avg_pop = st.slider("⭐ Artist Avg Popularity", 
                        int(get_range("avg_artist_popularity", "min", 0)), 
                        int(get_range("avg_artist_popularity", "max", 100)), 
                        int(get_range("avg_artist_popularity", "mean", 48)))

st.write("")
st.write("")
st.markdown("#### 🎼 Structure")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    key = st.selectbox("Musical Key", [0,1,2,3,4,5,6,7,8,9,10,11], help="0=C, 1=C#, 2=D...")
with col_s2:
    mode = st.radio("Major / Minor", [1, 0], format_func=lambda x: "Major" if x==1 else "Minor", horizontal=True)
with col_s3:
    st.markdown("#### 🎼 Structure")
    duration_m = st.number_input("Duration (mins)", 
                                 get_range("duration_ms", "min", 15000.0) / 60000.0, 
                                 get_range("duration_ms", "max", 5000000.0) / 60000.0, 
                                 get_range("duration_ms", "mean", 237411.0) / 60000.0)
    duration_ms = int(duration_m * 60 * 1000)

year = 2024 # Auto-assume current year for new songs

# ==========================================
# PREDICTION ENGINE
# ==========================================
st.write("---")

if st.button("🔮 Predict Hit Potential"):
    
    with st.spinner("Analyzing deep audio features..."):
        
        # 1. Gather User Base Features
        song = {
            "danceability": danceability,
            "energy": energy,
            "loudness": loudness,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo,
            "duration_ms": duration_ms,
            "sentiment_score": sentiment_score,
            "total_artist_followers": followers,
            "avg_artist_popularity": avg_pop,
            "key": key,
            "mode": mode,
            "year": year
        }

        # 2. Auto-Compute Advanced Interaction Features
        song["energy_danceability"] = song["energy"] * song["danceability"]
        song["valence_energy"] = song["valence"] * song["energy"]
        song["tempo_energy"] = song["tempo"] * song["energy"]
        song["loudness_energy"] = song["loudness"] * song["energy"]
        song["speechiness_valence"] = song["speechiness"] * song["valence"]
        song["popularity_per_follower"] = song["avg_artist_popularity"] / (song["total_artist_followers"] + 1)
        song["acousticness_energy"] = song["acousticness"] * song["energy"]

        # 3. Format Exactly as Trained
        final_list = [song[f] for f in feature_names]
        song_df = pd.DataFrame([final_list], columns=feature_names)
        
        # 4. Scale and Predict
        scaled_song = scaler.transform(song_df)
        prob = model.predict(scaled_song)[0][0]

        # 5. Display Results
        st.write("")
        st.write("")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="AI Confidence Score", value=f"{prob*100:.1f}%")
            st.caption(f"Requires exactly {best_thresh*100:.1f}% to be considered a hit")
            
        with res_col2:
            if prob >= best_thresh:
                st.success("✅ GLOBAL HIT Dashboard predicts this track will chart based on audio signatures.")
            else:
                st.error("❌ FLOP Dashboard calculates this track lacks the specific audio signatures of top 10% hits.")
# ==========================================
# TRAINING METRICS DASHBOARD
# ==========================================
st.write("---")
st.write("### 📊 Training Metrics & Model Integrity")
st.caption("Official performance data evaluated on the holdout test set.")

if df_metrics is not None and roc_data is not None:
    
    tab1, tab2 = st.tabs(["🏆 Performance Table", "📈 Integrity Curves (ROC)"])
    
    with tab1:
        # Style the dataframe so the DNN row pops out
        def highlight_dnn(s):
            return ['background-color: rgba(29, 185, 84, 0.2); font-weight: bold' if s.Model.startswith("DNN") else '' for v in s]
        
        st.dataframe(
            df_metrics.style.apply(highlight_dnn, axis=1).format({
                "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", 
                "F1-Score": "{:.3f}", "AUC-ROC": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        st.caption("*Note: Baseline models skew towards 'Non-Hit' due to dataset imbalance. The DNN utilizes confidence-weighted evaluation to isolate precise Hit detection capacity.*")
        
    with tab2:
        fig = go.Figure()
        
        colors = {"LR": "gray", "RF": "orange", "XGB": "purple", "DNN": "#1DB954"}
        
        for name, data in roc_data.items():
            width = 4 if name == "DNN" else 2
            dash = "solid" if name == "DNN" else "dot"
            
            fig.add_trace(go.Scatter(
                x=data["fpr"], y=data["tpr"],
                mode='lines',
                name=f"{name} (AUC: {data['auc']:.3f})",
                line=dict(color=colors.get(name, "white"), width=width, dash=dash)
            ))
            
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Random Chance", line=dict(color="gray", dash="dash")))
        
        fig.update_layout(
            title="Receiver Operating Characteristic (ROC)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Metrics data not found. Please re-run the updated Kaggle script and download `metrics_df.pkl` and `roc_data.pkl` to the models folder.")
