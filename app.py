import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
from datetime import datetime
import plotly.express as px

# =========================
# SENTIMENT ANALYSIS CONFIG
# =========================
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🧠",
    layout="wide"
)

BASE = Path(__file__).parent
MODEL = BASE / "sentiment_model.pkl"
VECTORIZER = BASE / "tfidf_vectorizer.pkl"

# =========================
# STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.stApp {background:#0e1117}
.block-container {padding-top:2rem}
.card {
    padding:18px;
    border-radius:14px;
    border:1px solid #30363d;
    background:#161b22;
    margin-bottom:15px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if not MODEL.exists() or not VECTORIZER.exists():
        return None, None
    return joblib.load(MODEL), joblib.load(VECTORIZER)

model, vectorizer = load_model()

if model is None:
    st.error("Sentiment Analysis model files not found.")
    st.info("Run train_model.py first.")
    st.stop()

# =========================
# FUNCTIONS
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def analyze(text):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])

    prediction = str(model.predict(features)[0]).strip().title()

    if hasattr(model, "predict_proba"):
        values = model.predict_proba(features)[0]
        classes = [str(x).strip().title() for x in model.classes_]
        probabilities = dict(zip(classes, values * 100))
        confidence = max(values) * 100
    else:
        probabilities = {}
        confidence = 0

    return prediction, confidence, probabilities


ASPECTS = {
    "🚚 Delivery": [
        "delivery", "delivered", "shipping", "shipment",
        "courier", "arrived", "delay", "late"
    ],
    "📦 Packaging": [
        "packaging", "package", "packing", "packed",
        "box", "wrapped", "parcel"
    ],
    "🎧 Customer Support": [
        "support", "customer service", "agent",
        "complaint", "refund", "help", "response"
    ],
    "🛍️ Product Quality": [
        "product", "quality", "material", "build",
        "design", "performance", "durable"
    ]
}


def detect_aspects(text):
    text = text.lower()
    return {
        aspect: any(word in text for word in words)
        for aspect, words in ASPECTS.items()
    }


def rating_match(rating, sentiment):
    sentiment = sentiment.lower()

    if rating >= 4 and "negative" in sentiment:
        return False
    if rating <= 2 and "positive" in sentiment:
        return False
    return True


def insight(sentiment, rating):
    if "positive" in sentiment.lower() and rating >= 4:
        return "Customer shows strong satisfaction and positive experience."
    if "negative" in sentiment.lower() and rating <= 2:
        return "Customer dissatisfaction is clearly reflected in both review and rating."
    if "neutral" in sentiment.lower():
        return "Customer feedback appears balanced or neutral."
    return "Rating and written feedback show different signals."

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("🧠 Sentiment Analysis")
    st.caption("Customer Experience Intelligence")
    st.divider()

    page = st.radio(
        "Sentiment Analysis",
        ["📝 Review Analysis", "📁 Bulk Analysis"]
    )

    st.divider()
    st.write("🟢 **Sentiment Analysis Engine:** Online")
    st.write(f"📊 Reviews: **{len(st.session_state.history)}**")

# =========================
# HEADER
# =========================
st.title("🧠 Sentiment Analysis")
st.caption(
    "AI-powered customer review, rating and experience analysis"
)

# =========================================================
# SINGLE REVIEW
# =========================================================
if page == "📝 Review Analysis":

    left, right = st.columns([2, 1])

    with left:
        st.subheader("💬 Customer Review")
        review = st.text_area(
            "Review",
            placeholder="Share your customer experience...",
            height=160,
            label_visibility="collapsed"
        )

    with right:
        st.subheader("⭐ Rating")
        rating = st.slider("Customer Rating", 1, 5, 5)
        st.write(f"Selected Rating: {'⭐' * rating}")

        c1, c2 = st.columns(2)
        c1.metric("Words", len(review.split()))
        c2.metric("Characters", len(review))

    if st.button(
        "🚀 Run Sentiment Analysis",
        type="primary",
        use_container_width=True
    ):
        if not review.strip():
            st.warning("Please enter a customer review.")
        else:
            sentiment, confidence, probabilities = analyze(review)

            st.session_state.result = {
                "sentiment": sentiment,
                "confidence": confidence,
                "probabilities": probabilities,
                "rating": rating,
                "aspects": detect_aspects(review),
                "match": rating_match(rating, sentiment),
                "insight": insight(sentiment, rating)
            }

            st.session_state.history.insert(0, {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Review": review[:35],
                "Rating": f"{rating}/5",
                "Sentiment": sentiment,
                "Confidence": f"{confidence:.1f}%"
            })

            st.session_state.history = st.session_state.history[:10]

    # =========================
    # RESULT
    # =========================
    if st.session_state.result:
        r = st.session_state.result

        st.divider()
        st.subheader("📊 Sentiment Analysis Results")

        a, b, c = st.columns(3)

        with a:
            if "positive" in r["sentiment"].lower():
                st.success(f"😊 {r['sentiment']}")
            elif "negative" in r["sentiment"].lower():
                st.error(f"😞 {r['sentiment']}")
            else:
                st.warning(f"😐 {r['sentiment']}")

        b.metric("🎯 Confidence", f"{r['confidence']:.1f}%")
        c.metric("⭐ Rating", f"{r['rating']}/5")

        # =========================
        # PROBABILITY
        # =========================
        st.subheader("📈 Sentiment Probability Distribution")

        if r["probabilities"]:
            pdf = pd.DataFrame({
                "Sentiment": list(r["probabilities"]),
                "Probability": list(r["probabilities"].values())
            })

            fig = px.bar(
                pdf,
                x="Sentiment",
                y="Probability",
                text_auto=".1f"
            )

            fig.update_layout(
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig, use_container_width=True)

        # =========================
        # CUSTOMER EXPERIENCE
        # =========================
        st.subheader("🔍 Customer Experience Analysis")

        aspect_cols = st.columns(2)

        for i, (aspect, mentioned) in enumerate(r["aspects"].items()):
            with aspect_cols[i % 2]:
                status = "Mentioned ✓" if mentioned else "Not Mentioned"
                if st.button(
                    f"{aspect} • {status}",
                    key=f"aspect_{i}",
                    use_container_width=True
                ):
                    st.session_state["selected"] = aspect

        # =========================
        # EXPERIENCE SHARING
        # =========================
        selected = st.session_state.get("selected")

        if selected:
            st.divider()
            st.subheader(f"{selected} Experience")

            experience = st.text_area(
                "Share your experience",
                placeholder=f"Tell us about your {selected.lower()} experience...",
                key="experience"
            )

            if st.button(
                f"🔎 Analyze {selected} Sentiment",
                type="primary",
                use_container_width=True
            ):
                if experience.strip():
                    s, conf, probs = analyze(experience)

                    x, y = st.columns(2)

                    with x:
                        if "positive" in s.lower():
                            st.success(f"😊 {s}")
                        elif "negative" in s.lower():
                            st.error(f"😞 {s}")
                        else:
                            st.warning(f"😐 {s}")

                    with y:
                        st.metric("Confidence", f"{conf:.1f}%")
                else:
                    st.warning("Please share your experience first.")

        # =========================
        # INSIGHTS
        # =========================
        st.divider()
        st.subheader("💡 Customer Insights")
        st.info(r["insight"])

        if r["match"]:
            st.success("✓ Rating and Sentiment Analysis appear consistent.")
        else:
            st.warning(
                "⚠️ Rating and Sentiment Analysis show a possible mismatch."
            )

# =========================================================
# BULK ANALYSIS
# =========================================================
else:
    st.subheader("📁 Bulk Sentiment Analysis")

    file = st.file_uploader(
        "Upload Customer Reviews CSV",
        type=["csv"]
    )

    if file:
        df = pd.read_csv(file)

        st.success(f"{len(df):,} reviews loaded.")

        column = st.selectbox(
            "Select Review / Comment Column",
            df.columns
        )

        if st.button(
            "🚀 Run Bulk Sentiment Analysis",
            type="primary",
            use_container_width=True
        ):
            texts = df[column].fillna("").astype(str)
            features = vectorizer.transform(texts)

            df["Sentiment"] = model.predict(features)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)
                df["Confidence (%)"] = np.max(probs, axis=1) * 100

            st.success("Bulk Sentiment Analysis completed.")

            counts = df["Sentiment"].value_counts()

            fig = px.pie(
                values=counts.values,
                names=counts.index,
                hole=.45,
                title="Sentiment Analysis Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df, use_container_width=True)

            st.download_button(
                "📥 Download Sentiment Analysis Report",
                df.to_csv(index=False).encode(),
                "sentiment_analysis_results.csv",
                "text/csv",
                use_container_width=True
            )

# =========================
# HISTORY
# =========================
if st.session_state.history and page == "📝 Review Analysis":
    st.divider()
    st.subheader("🕘 Recent Sentiment Analysis")
    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True,
        hide_index=True
    )

# =========================
# FOOTER
# =========================
st.divider()
st.caption(
    "🧠 Sentiment Analysis • Customer Experience Intelligence • "
    "TF-IDF + Machine Learning"
)
