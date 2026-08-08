import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# =========================================================
# SENTIMENT ANALYSIS - PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"


# =========================================================
# SESSION STATE MANAGEMENT
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "selected_aspect" not in st.session_state:
    st.session_state.selected_aspect = None

if "result" not in st.session_state:
    st.session_state.result = None


# =========================================================
# ADVANCED UI/UX CUSTOM STYLING (CSS)
# =========================================================

st.markdown("""
<style>
    /* Main App Background & Typography */
    .stApp {
        background-color: #0b0f19;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Sidebar Custom Styling */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    /* Global Headings */
    h1, h2, h3 {
        color: #f9fafb !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }

    /* Custom Metric Card Container */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="stMetric"] label {
        color: #9ca3af !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f3f4f6 !important;
        font-weight: 700 !important;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #1f2937 !important;
        color: #f9fafb !important;
        border: 1px solid #374151 !important;
        border-radius: 10px !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }

    /* Button Enhancements */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }

    /* Aspect Badges / Custom Cards */
    .aspect-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Custom Header Banner */
    .header-banner {
        background: linear-gradient(90deg, #3b82f6 0%, #6366f1 50%, #8b5cf6 100%);
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
    }
    
    .header-banner h1 {
        margin: 0;
        font-size: 2.2rem;
        color: white !important;
    }

    .header-banner p {
        margin-top: 6px;
        margin-bottom: 0;
        opacity: 0.9;
        font-size: 1rem;
    }

    /* Footer Styling */
    .footer {
        text-align: center;
        padding: 20px 0;
        margin-top: 40px;
        border-top: 1px solid #1f2937;
        color: #6b7280;
        font-size: 0.85rem;
    }

    .footer span {
        color: #8b5cf6;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SENTIMENT ANALYSIS - MODEL LOADER
# =========================================================

@st.cache_resource
def load_sentiment_model():
    if not MODEL_PATH.exists():
        return None, None, "Sentiment Analysis model (sentiment_model.pkl) not found."

    if not VECTORIZER_PATH.exists():
        return None, None, "Sentiment Analysis Vectorizer (tfidf_vectorizer.pkl) not found."

    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer, None
    except Exception as error:
        return None, None, str(error)


model, vectorizer, model_error = load_sentiment_model()

if model is None:
    st.error("⚠️ Critical Error: Sentiment Analysis Engine failed to initialize.")
    st.code(model_error)
    st.stop()


# =========================================================
# SENTIMENT ANALYSIS PREDICTION CORE
# =========================================================

def predict_sentiment(text):
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    sentiment = str(prediction).strip().title()
    probabilities = {}

    if hasattr(model, "predict_proba"):
        values = model.predict_proba(features)[0]
        if hasattr(model, "classes_"):
            probabilities = {
                str(label).strip().title(): float(value) * 100
                for label, value in zip(model.classes_, values)
            }
        confidence = float(np.max(values) * 100)
    else:
        confidence = None

    return sentiment, confidence, probabilities


# =========================================================
# ASPECT DEFINITIONS & DETECTION
# =========================================================

ASPECTS = {
    "🚚 Delivery": "Delivery",
    "📦 Packaging": "Packaging",
    "🎧 Customer Support": "Customer Support",
    "🛍️ Product Quality": "Product Quality"
}

ASPECT_KEYWORDS = {
    "Delivery": ["delivery", "delivered", "deliver", "shipping", "shipment", "courier", "arrived", "late", "delay"],
    "Packaging": ["packaging", "package", "packing", "packed", "box", "wrapped", "damaged package", "damaged box"],
    "Customer Support": ["support", "customer service", "helpdesk", "agent", "representative", "complaint", "refund", "response"],
    "Product Quality": ["product", "quality", "material", "build", "performance", "design", "durable", "value"]
}


def detect_aspects(text):
    text_lower = text.lower()
    result = {}
    for display_name, aspect_name in ASPECTS.items():
        found = any(keyword in text_lower for keyword in ASPECT_KEYWORDS[aspect_name])
        result[display_name] = "Mentioned" if found else "Not Mentioned"
    return result


def create_sentiment_insight(sentiment, rating):
    sentiment_lower = sentiment.lower()
    if "positive" in sentiment_lower:
        if rating >= 4:
            return "Sentiment Analysis Insight: Customer feedback demonstrates high satisfaction and alignment with rating."
        return "Sentiment Analysis Insight: The written text reveals positive sentiment despite a conservative numerical rating."
    elif "negative" in sentiment_lower:
        if rating <= 2:
            return "Sentiment Analysis Insight: Explicit customer dissatisfaction is confirmed across text and numeric rating."
        return "Sentiment Analysis Insight: Negative sentiment detected in text despite a high numerical rating."
    return "Sentiment Analysis Insight: The feedback displays balanced or neutral overall sentiment."


# =========================================================
# SIDEBAR NAVIGATION & SYSTEM STATUS
# =========================================================

with st.sidebar:
    st.markdown("## 🧠 Sentiment Analysis")
    st.caption("AI-Powered Customer Feedback Intelligence")

    st.divider()

    page = st.radio(
        "Sentiment Analysis Navigation",
        [
            "📝 Single Review Sentiment Analysis",
            "📁 Bulk Review Sentiment Analysis"
        ],
        index=0
    )

    st.divider()

    st.markdown("### ⚡ System Status")
    st.markdown("🟢 **Sentiment Engine:** Online")
    st.markdown("📌 **Model:** TF-IDF + Classifier")
    st.markdown(f"📊 **Reviews Analyzed:** `{len(st.session_state.history)}`")

    st.divider()

    st.caption("Machine Learning • NLP • Sentiment Analysis System v2.0")


# =========================================================
# HEADER BANNER
# =========================================================

st.markdown("""
<div class="header-banner">
    <h1>📊 Sentiment Analysis
    <p>Analyze text sentiment, detect operational aspects, and extract business insights in real time.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MODULE 1: SINGLE REVIEW SENTIMENT ANALYSIS
# =========================================================

if page == "📝 Single Review Sentiment Analysis":

    col_input, col_meta = st.columns([1.8, 1], gap="large")

    with col_input:
        st.markdown("### 📝 Enter Review for Sentiment Analysis")
        review = st.text_area(
            "Customer Review Text",
            placeholder="Type or paste a customer review here... (e.g., 'The product quality is superb, but customer support was unhelpful and delivery took over a week.')",
            height=180,
            label_visibility="collapsed"
        )

    with col_meta:
        st.markdown("### ⭐ Customer Rating")
        rating = st.slider(
            "Star Rating",
            min_value=1,
            max_value=5,
            value=5,
            help="Select the numerical rating attached to the review."
        )

        st.markdown(f"**Selected Rating:** {'⭐' * rating}")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Word Count", len(review.split()))
        with m_col2:
            st.metric("Character Count", len(review))

    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("🚀 Run Sentiment Analysis", type="primary", use_container_width=True)

    # -----------------------------------------------------
    # PROCESSING SENTIMENT ANALYSIS
    # -----------------------------------------------------

    if analyze:
        if not review.strip():
            st.warning("⚠️ Please provide a text review before running Sentiment Analysis.")
        else:
            with st.spinner("Analyzing text sentiment and extracting insights..."):
                sentiment, confidence, probabilities = predict_sentiment(review)
                detected_aspects = detect_aspects(review)
                insight = create_sentiment_insight(sentiment, rating)

            mismatch = (rating >= 4 and "negative" in sentiment.lower()) or (
                rating <= 2 and "positive" in sentiment.lower()
            )

            st.session_state.result = {
                "sentiment": sentiment,
                "confidence": confidence,
                "probabilities": probabilities,
                "rating": rating,
                "aspects": detected_aspects,
                "insight": insight,
                "mismatch": mismatch
            }

            st.session_state.history.insert(
                0,
                {
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Review Preview": review[:40] + "..." if len(review) > 40 else review,
                    "Rating": f"{rating} ⭐",
                    "Sentiment": sentiment,
                    "Confidence": f"{confidence:.1f}%" if confidence else "N/A"
                }
            )

            st.session_state.history = st.session_state.history[:10]

    # =====================================================
    # DISPLAY SENTIMENT ANALYSIS RESULTS
    # =====================================================

    if st.session_state.result:
        result = st.session_state.result

        st.divider()
        st.markdown("## 📊 Sentiment Analysis Results")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            if "positive" in result["sentiment"].lower():
                st.success(f"### 😊 Sentiment: {result['sentiment']}")
            elif "negative" in result["sentiment"].lower():
                st.error(f"### 😞 Sentiment: {result['sentiment']}")
            else:
                st.warning(f"### 😐 Sentiment: {result['sentiment']}")

        with res_col2:
            st.metric(
                "🎯 Sentiment Analysis Confidence",
                f"{result['confidence']:.2f}%" if result["confidence"] else "N/A"
            )

        with res_col3:
            st.metric(
                "⭐ Customer Star Rating",
                f"{result['rating']} / 5"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------
        # PROBABILITY DISTRIBUTION CHART
        # -------------------------------------------------

        st.markdown("### 📈 Sentiment Analysis Probability Distribution")

        if result["probabilities"]:
            prob_df = pd.DataFrame({
                "Sentiment Class": list(result["probabilities"].keys()),
                "Probability (%)": list(result["probabilities"].values())
            })

            fig = px.bar(
                prob_df,
                x="Sentiment Class",
                y="Probability (%)",
                text="Probability (%)",
                color="Sentiment Class",
                color_discrete_map={
                    "Positive": "#10b981",
                    "Negative": "#ef4444",
                    "Neutral": "#f59e0b"
                }
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                marker_line_color="#1f2937",
                marker_line_width=1.5
            )

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f9fafb"),
                height=320,
                xaxis=dict(title=""),
                yaxis=dict(title="Probability (%)", range=[0, 115], showgrid=True, gridcolor="#1f2937"),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------
        # ASPECT-BASED SENTIMENT ANALYSIS
        # -------------------------------------------------

        st.markdown("### 🔍 Aspect-Based Sentiment Analysis")
        st.caption("Identify specific feedback categories and analyze sentiment per aspect.")

        grid_col1, grid_col2 = st.columns(2)
        aspect_items = list(result["aspects"].items())

        for idx, (aspect, status) in enumerate(aspect_items):
            target_col = grid_col1 if idx % 2 == 0 else grid_col2

            with target_col:
                is_mentioned = status == "Mentioned"
                badge = "✓ Detected" if is_mentioned else "+ Add Aspect Feedback"
                button_label = f"{aspect}  •  {badge}"

                if st.button(button_label, key=f"aspect_btn_{idx}", use_container_width=True):
                    st.session_state.selected_aspect = aspect

        # -------------------------------------------------
        # DRILL-DOWN ASPECT SENTIMENT ANALYSIS
        # -------------------------------------------------

        if st.session_state.selected_aspect:
            selected = st.session_state.selected_aspect
            aspect_name = ASPECTS[selected]

            st.divider()
            st.markdown(f"### 🔍 Detailed Sentiment Analysis: {selected}")

            exp_text = st.text_area(
                f"Describe your specific {aspect_name} feedback:",
                placeholder=f"Provide specific details regarding {aspect_name.lower()}...",
                height=120,
                key=f"exp_{aspect_name}"
            )

            if st.button(f"🔎 Run Sentiment Analysis for {aspect_name}", type="primary", use_container_width=True):
                if not exp_text.strip():
                    st.warning(f"Please enter text to analyze sentiment for {aspect_name}.")
                else:
                    with st.spinner(f"Analyzing Sentiment for {aspect_name}..."):
                        exp_sent, exp_conf, exp_probs = predict_sentiment(exp_text)

                    c_a, c_b = st.columns(2)
                    with c_a:
                        if "positive" in exp_sent.lower():
                            st.success(f"**{aspect_name} Sentiment:** {exp_sent} 😊")
                        elif "negative" in exp_sent.lower():
                            st.error(f"**{aspect_name} Sentiment:** {exp_sent} 😞")
                        else:
                            st.warning(f"**{aspect_name} Sentiment:** {exp_sent} 😐")

                    with c_b:
                        st.metric(f"{aspect_name} Confidence", f"{exp_conf:.2f}%" if exp_conf else "N/A")

        # -------------------------------------------------
        # INSIGHTS & CONSISTENCY CHECK
        # -------------------------------------------------

        st.divider()
        st.markdown("### 💡 Sentiment Analysis Insights & Consistency")

        st.info(result["insight"])

        if result["mismatch"]:
            st.warning("⚠️ **Sentiment Mismatch Alert:** Numerical rating and calculated text sentiment appear inconsistent.")
        else:
            st.success("✓ **Sentiment Consistency Verified:** Numerical rating aligns with text Sentiment Analysis.")


# =========================================================
# RECENT SENTIMENT ANALYSIS HISTORY
# =========================================================

if st.session_state.history:
    st.divider()
    st.markdown("### 🕘 Recent Sentiment Analysis Log")
    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# MODULE 2: BULK REVIEW SENTIMENT ANALYSIS
# =========================================================

if page == "📁 Bulk Review Sentiment Analysis":

    st.markdown("## 📁 Bulk Sentiment Analysis Engine")
    st.caption("Upload datasets to perform batch Sentiment Analysis with export capabilities.")

    uploaded_file = st.file_uploader(
        "Upload Customer Reviews CSV File",
        type=["csv"],
        help="Upload a CSV file containing customer reviews or comments."
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✓ Dataset loaded successfully: `{len(df):,}` rows found.")

        text_column = st.selectbox(
            "Select Review / Comment Column for Sentiment Analysis",
            df.columns
        )

        if st.button("🚀 Execute Bulk Sentiment Analysis", type="primary", use_container_width=True):
            with st.spinner("Processing dataset through Sentiment Analysis Engine..."):
                texts = df[text_column].fillna("").astype(str)
                features = vectorizer.transform(texts)
                
                # Model predictions
                df["Predicted Sentiment"] = model.predict(features)

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)
                    df["Sentiment Confidence (%)"] = np.max(probs, axis=1) * 100

            st.success("🎉 Bulk Sentiment Analysis completed successfully!")

            # -------------------------------------------------
            # BULK SUMMARY METRICS
            # -------------------------------------------------

            b_m1, b_m2, b_m3 = st.columns(3)
            
            sentiment_counts = df["Predicted Sentiment"].value_counts()
            
            with b_m1:
                st.metric("Total Reviews Analyzed", len(df))
            with b_m2:
                top_sentiment = sentiment_counts.index[0] if not sentiment_counts.empty else "N/A"
                st.metric("Dominant Sentiment", str(top_sentiment).title())
            with b_m3:
                avg_conf = df["Sentiment Confidence (%)"].mean() if "Sentiment Confidence (%)" in df.columns else 0
                st.metric("Average Confidence", f"{avg_conf:.1f}%")

            # -------------------------------------------------
            # BULK DISTRIBUTION CHART
            # -------------------------------------------------

            st.markdown("### 📊 Bulk Sentiment Analysis Distribution")
            
            fig_bulk = px.pie(
                df, 
                names="Predicted Sentiment", 
                title="Sentiment Analysis Breakdown",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_bulk.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f9fafb")
            )
            st.plotly_chart(fig_bulk, use_container_width=True)

            # -------------------------------------------------
            # DATASET PREVIEW & DOWNLOAD
            # -------------------------------------------------

            st.markdown("### 📋 Sentiment Analysis Result Dataset")
            st.dataframe(df, use_container_width=True)

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Sentiment Analysis Results (CSV)",
                data=csv_data,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
                use_container_width=True
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    Sentiment Analysis Platform • Powered by <span>Streamlit</span> & <span>Scikit-Learn</span> • Sentiment Analysis Engine v2.0
</div>
""", unsafe_allow_html=True)