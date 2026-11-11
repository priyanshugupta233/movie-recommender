import os
import pandas as pd
import requests
import scipy.sparse as sp
import streamlit as st
import streamlit_analytics2 as streamlit_analytics
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Bollywood & Hollywood Recommender",
                   page_icon="🎬", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = "https://image.tmdb.org/t/p/w342"


# ---------- Load model (once) ----------
@st.cache_resource
def load_data():
    movies = pd.read_pickle(os.path.join(BASE, "movies.pkl"))
    matrix = sp.load_npz(os.path.join(BASE, "tfidf_matrix.npz")).tocsr()
    return movies, matrix

movies, tfidf_matrix = load_data()
label_to_idx = pd.Series(movies.index, index=movies["label"])


# ---------- API key (env var or Streamlit secrets) ----------
def get_api_key():
    key = os.environ.get("TMDB_API_KEY")
    if key:
        return key
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return None

api_key = get_api_key()


# ---------- Poster fetching (cached for 24h) ----------
@st.cache_data(ttl=86400, show_spinner=False)
def get_poster(tmdb_id, imdb_id, key):
    if not key:
        return None
    try:
        if pd.notna(tmdb_id):
            r = requests.get(f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
                             params={"api_key": key}, timeout=5)
            path = r.json().get("poster_path")
        elif pd.notna(imdb_id):
            r = requests.get(f"https://api.themoviedb.org/3/find/{imdb_id}",
                             params={"api_key": key, "external_source": "imdb_id"},
                             timeout=5)
            found = r.json().get("movie_results", [])
            path = found[0].get("poster_path") if found else None
        else:
            return None
        return IMG + path if path else None
    except Exception:
        return None


# ---------- Recommender ----------
def recommend(label, n=5, industry="All"):
    i = label_to_idx[label]
    scores = linear_kernel(tfidf_matrix[i], tfidf_matrix).ravel()
    scores[i] = -1
    if industry == "Same as selected":
        industry = movies.loc[i, "industry"]
    if industry != "All":
        scores[(movies["industry"] != industry).values] = -1
    top = [t for t in scores.argsort()[::-1][:n] if scores[t] > 0.03]
    out = movies.loc[top].copy()
    out["score"] = scores[top]
    return out


# ---------- UI ----------
streamlit_analytics.start_tracking()
st.title("🎬 Bollywood & Hollywood Movie Recommender")
st.caption("Content-based: matches genres, cast, directors and plot keywords using TF-IDF.")

with st.sidebar:
    st.header("Settings")
    industry = st.radio("Recommend from",
                        ["All", "Bollywood", "Hollywood", "Same as selected"])
    n = st.slider("How many recommendations?", 3, 10, 5)
    if not api_key:
        st.info("No TMDB API key found, so posters are hidden.")

labels = sorted(movies["label"])
default = labels.index("Inception (2010)") if "Inception (2010)" in labels else 0
choice = st.selectbox("🔍 Pick (or type) a movie you like", labels, index=default)

# Selected movie details
sel = movies.loc[label_to_idx[choice]]
c1, c2 = st.columns([1, 4])
with c1:
    p = get_poster(sel["tmdb_id"], sel["imdb_id"], api_key)
    if p:
        st.image(p)
with c2:
    st.subheader(choice)
    st.write(f"**Industry:** {sel['industry']}")
    st.write(f"**Genres:** {', '.join(sel['genres']) or 'N/A'}")
    st.write(f"**Director:** {', '.join(sel['director']) or 'N/A'}")
    st.write(f"**Cast:** {', '.join(sel['cast']) or 'N/A'}")
    if sel["overview"]:
        st.write(sel["overview"])

st.divider()
st.subheader("You might also like")

recs = recommend(choice, n, industry)

if recs.empty:
    st.warning("No strong matches found. Try a different filter.")
else:
    if industry not in ("All", "Same as selected") and industry != sel["industry"]:
        st.caption("ℹ️ Cross-industry picks are based mostly on shared genres.")
    per_row = 5
    for start in range(0, len(recs), per_row):
        row = recs.iloc[start:start + per_row]
        cols = st.columns(per_row)
        for col, (_, r) in zip(cols, row.iterrows()):
            with col:
                poster = get_poster(r["tmdb_id"], r["imdb_id"], api_key)
                if poster:
                    st.image(poster)
                else:
                    st.markdown("## 🎞️")
                st.markdown(f"**{r['label']}**")
                st.caption(f"{r['industry']} • similarity {r['score']:.2f}")
                st.caption(", ".join(r["genres"][:3]))


st.divider()
st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")

def _secret(name):
    val = os.environ.get(name)
    if val:
        return val
    try:
        return st.secrets[name]
    except Exception:
        return None

streamlit_analytics.stop_tracking(unsafe_password=_secret("ANALYTICS_PASSWORD"))
