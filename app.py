import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix EDA Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    background-color: #0d0d0d;
    color: #e8e8e8;
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161616 !important;
    border-right: 1px solid #2a2a2a;
}
[data-testid="stSidebar"] * { color: #e8e8e8 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 16px;
}
[data-testid="stMetricValue"] { color: #e50914 !important; font-size: 2rem !important; }

/* Headers */
h1 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 3px; color: #e50914 !important; font-size: 3rem !important; }
h2 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 2px; color: #ffffff !important; font-size: 1.8rem !important; border-bottom: 2px solid #e50914; padding-bottom: 6px; }
h3 { color: #cccccc !important; }

/* Tabs */
[data-baseweb="tab-list"] { background: #1a1a1a; border-radius: 8px; padding: 4px; }
[data-baseweb="tab"] { color: #888 !important; }
[aria-selected="true"] { color: #e50914 !important; background: #2a2a2a !important; border-radius: 6px !important; }

/* Selectbox / slider labels */
label { color: #aaaaaa !important; font-size: 0.85rem !important; }

/* Dividers */
hr { border-color: #2a2a2a !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #e50914; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#1a1a1a",
    "axes.facecolor":    "#1a1a1a",
    "axes.edgecolor":    "#2a2a2a",
    "axes.labelcolor":   "#cccccc",
    "xtick.color":       "#888888",
    "ytick.color":       "#888888",
    "text.color":        "#e8e8e8",
    "grid.color":        "#2a2a2a",
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
    "axes.titlecolor":   "#ffffff",
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
})
NETFLIX_RED  = "#e50914"
ACCENT_GOLD  = "#f5a623"
PALETTE_MAIN = [NETFLIX_RED, "#f5a623", "#4ecdc4", "#95e1d3", "#f38181", "#a29bfe", "#fd79a8", "#00b894", "#fdcb6e", "#636e72"]

# ── Load & cache data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df.drop_duplicates(inplace=True)

    # Drop fully-empty unnamed columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    df["country"]  = df["country"].fillna("Unknown")
    df["director"] = df["director"].fillna("Unknown")
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month

    # Movie duration in minutes
    movie_mask = df["type"] == "Movie"
    df.loc[movie_mask, "duration_min"] = (
        df.loc[movie_mask, "duration"].str.replace(" min", "", regex=False).astype(float, errors="ignore")
    )

    # TV show seasons
    tv_mask = df["type"] == "TV Show"
    df.loc[tv_mask, "seasons"] = (
        df.loc[tv_mask, "duration"]
          .str.replace(" Season", "", regex=False)
          .str.replace("s", "", regex=False)
          .astype(float, errors="ignore")
    )
    return df

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Netflix EDA")
    st.markdown("---")

    uploaded = st.file_uploader("Upload `netflix_titles.csv`", type="csv")
    if uploaded:
        df = load_data(uploaded)
        st.success(f"✅ {len(df):,} titles loaded")
    else:
        st.info("Upload your CSV to begin — or a sample view is shown below.")
        # Build a tiny demo so the dashboard isn't empty on first load
        df = None

    st.markdown("---")
    if df is not None:
        content_filter = st.multiselect(
            "Content Type",
            options=df["type"].unique().tolist(),
            default=df["type"].unique().tolist(),
        )
        year_range = st.slider(
            "Release Year",
            int(df["release_year"].min()),
            int(df["release_year"].max()),
            (int(df["release_year"].min()), int(df["release_year"].max())),
        )
        top_n = st.slider("Top-N for rankings", 5, 20, 10)

        # Apply filters
        df = df[
            df["type"].isin(content_filter) &
            df["release_year"].between(*year_range)
        ]
        st.markdown(f"**{len(df):,}** titles after filters")

# ── If no data, show upload prompt ────────────────────────────────────────────
if df is None:
    st.markdown("# NETFLIX ANALYTICS DASHBOARD")
    st.markdown("### Upload your `netflix_titles.csv` in the sidebar to explore the data.")
    st.markdown("""
    **What you'll get:**
    - 📊 KPI summary cards  
    - 🗺️ Country & genre breakdowns  
    - 📅 Content growth over time  
    - ⏱️ Movie duration & TV season distributions  
    - 🔥 Genre × Rating heatmap  
    """)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# NETFLIX ANALYTICS DASHBOARD")

# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Titles",    f"{len(df):,}")
k2.metric("Movies",          f"{(df['type']=='Movie').sum():,}")
k3.metric("TV Shows",        f"{(df['type']=='TV Show').sum():,}")
k4.metric("Countries",       f"{df['country'].nunique():,}")
k5.metric("Unique Genres",   f"{df['listed_in'].str.split(', ').explode().nunique():,}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🌍 Countries & Genres",
    "📅 Timeline",
    "⏱️ Durations",
    "🔥 Heatmap",
])

# ══════════════════════════════════════════════════════
# TAB 1  –  OVERVIEW
# ══════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    # Movies vs TV Shows donut
    with col1:
        st.markdown("## Content Split")
        counts = df["type"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=[NETFLIX_RED, ACCENT_GOLD],
            wedgeprops=dict(width=0.55, edgecolor="#0d0d0d", linewidth=2),
            textprops={"color": "#e8e8e8", "fontsize": 11},
        )
        for at in autotexts:
            at.set_color("#0d0d0d"); at.set_fontweight("bold")
        ax.set_title("Movies vs TV Shows", pad=12)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Rating distribution
    with col2:
        st.markdown("## Rating Distribution")
        rating_counts = df["rating"].value_counts().dropna().head(top_n)
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.barh(
            rating_counts.index[::-1],
            rating_counts.values[::-1],
            color=PALETTE_MAIN[:len(rating_counts)],
            edgecolor="none",
            height=0.65,
        )
        ax.set_xlabel("Count")
        ax.set_title("Content Ratings")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        for bar in bars:
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    f"{int(bar.get_width()):,}", va="center", fontsize=8, color="#aaa")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Top directors table
    st.markdown("## Top Directors")
    directors = (
        df[df["director"] != "Unknown"]["director"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    directors.columns = ["Director", "Titles"]
    directors.index = directors.index + 1
    st.dataframe(
        directors.style.bar(subset=["Titles"], color=NETFLIX_RED, vmin=0),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════
# TAB 2  –  COUNTRIES & GENRES
# ══════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    # Top countries
    with col1:
        st.markdown("## Top Countries")
        country_counts = df["country"].value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(5, 5))
        bars = ax.barh(
            country_counts.index[::-1],
            country_counts.values[::-1],
            color=PALETTE_MAIN[:len(country_counts)],
            edgecolor="none",
            height=0.7,
        )
        ax.set_xlabel("Titles")
        ax.set_title(f"Top {top_n} Countries")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Top genres
    with col2:
        st.markdown("## Top Genres")
        genre_counts = (
            df["listed_in"].str.split(", ").explode().value_counts().head(top_n)
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(genre_counts)))
        bars = ax.barh(
            genre_counts.index[::-1],
            genre_counts.values[::-1],
            color=colors,
            edgecolor="none",
            height=0.7,
        )
        ax.set_xlabel("Titles")
        ax.set_title(f"Top {top_n} Genres")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Genres by top countries
    st.markdown("## Genre Mix — Top 5 Countries")
    top5 = df["country"].value_counts().head(5).index
    df_exp = (
        df[df["country"].isin(top5)]
        .assign(genre=lambda d: d["listed_in"].str.split(", "))
        .explode("genre")
    )
    top_genres_list = df_exp["genre"].value_counts().head(8).index
    df_exp2 = df_exp[df_exp["genre"].isin(top_genres_list)]
    pivot = df_exp2.groupby(["country", "genre"]).size().unstack(fill_value=0)
    pivot = pivot[top_genres_list]

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, color=PALETTE_MAIN[:len(top_genres_list)], edgecolor="none", width=0.7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Top Genres by Country")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8,
              facecolor="#1a1a1a", edgecolor="#2a2a2a", labelcolor="#ccc")
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════
# TAB 3  –  TIMELINE
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown("## Content Added Over Time")

    # By year & type
    yearly = (
        df.dropna(subset=["year_added"])
        .groupby(["year_added", "type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    yearly["year_added"] = yearly["year_added"].astype(int)

    fig, ax = plt.subplots(figsize=(11, 4))
    if "Movie" in yearly.columns:
        ax.fill_between(yearly["year_added"], yearly["Movie"], alpha=0.35, color=NETFLIX_RED, label="Movie")
        ax.plot(yearly["year_added"], yearly["Movie"], color=NETFLIX_RED, linewidth=2.5)
    if "TV Show" in yearly.columns:
        ax.fill_between(yearly["year_added"], yearly["TV Show"], alpha=0.35, color=ACCENT_GOLD, label="TV Show")
        ax.plot(yearly["year_added"], yearly["TV Show"], color=ACCENT_GOLD, linewidth=2.5)
    ax.set_xlabel("Year"); ax.set_ylabel("Titles Added")
    ax.set_title("Content Added per Year by Type")
    ax.legend(facecolor="#1a1a1a", edgecolor="#2a2a2a", labelcolor="#ccc")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Monthly heatmap of additions
    st.markdown("## Monthly Additions Heatmap")
    month_data = (
        df.dropna(subset=["year_added", "month_added"])
        .groupby(["year_added", "month_added"])
        .size()
        .unstack(fill_value=0)
    )
    month_data.index = month_data.index.astype(int)
    month_data.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(month_data.columns)]

    fig, ax = plt.subplots(figsize=(11, max(4, len(month_data) * 0.35)))
    sns.heatmap(
        month_data, ax=ax,
        cmap=sns.color_palette("RdYlGn", as_cmap=True),
        linewidths=0.4, linecolor="#0d0d0d",
        annot=True, fmt="d",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title("Titles Added per Month & Year")
    ax.set_ylabel("Year"); ax.set_xlabel("Month")
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════
# TAB 4  –  DURATIONS
# ══════════════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns(2)

    # Movie duration histogram
    with col1:
        st.markdown("## Movie Durations")
        movies = df[df["type"] == "Movie"].dropna(subset=["duration_min"])
        if not movies.empty:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(movies["duration_min"], bins=35, color=NETFLIX_RED, edgecolor="#0d0d0d", alpha=0.85)
            # KDE overlay
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(movies["duration_min"].dropna())
            xs = np.linspace(movies["duration_min"].min(), movies["duration_min"].max(), 200)
            ax2 = ax.twinx()
            ax2.plot(xs, kde(xs), color=ACCENT_GOLD, linewidth=2)
            ax2.set_yticks([])
            ax2.tick_params(colors="#0d0d0d")
            ax.set_xlabel("Duration (minutes)")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Movie Durations")
            median_val = movies["duration_min"].median()
            ax.axvline(median_val, color=ACCENT_GOLD, linestyle="--", linewidth=1.5, label=f"Median: {int(median_val)} min")
            ax.legend(facecolor="#1a1a1a", edgecolor="#2a2a2a", labelcolor="#ccc")
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption(f"Median: **{int(movies['duration_min'].median())} min** · Mean: **{movies['duration_min'].mean():.0f} min** · Std: **{movies['duration_min'].std():.0f} min**")
        else:
            st.warning("No movie data available with current filters.")

    # TV seasons bar chart
    with col2:
        st.markdown("## TV Show Seasons")
        tv = df[df["type"] == "TV Show"].dropna(subset=["seasons"])
        if not tv.empty:
            season_counts = tv["seasons"].value_counts().sort_index().head(12)
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(
                season_counts.index.astype(int),
                season_counts.values,
                color=PALETTE_MAIN[:len(season_counts)],
                edgecolor="#0d0d0d",
                width=0.6,
            )
            ax.set_xlabel("Number of Seasons")
            ax.set_ylabel("Number of Shows")
            ax.set_title("TV Shows by Season Count")
            ax.set_xticks(season_counts.index.astype(int))
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        str(int(bar.get_height())), ha="center", va="bottom", fontsize=8, color="#aaa")
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption(f"Most shows have **{int(tv['seasons'].mode()[0])} season** · Max: **{int(tv['seasons'].max())} seasons**")
        else:
            st.warning("No TV show data available with current filters.")

    # Box plot: movie duration by rating
    st.markdown("## Movie Duration by Rating")
    movies_r = df[(df["type"] == "Movie")].dropna(subset=["duration_min", "rating"])
    top_ratings = movies_r["rating"].value_counts().head(8).index
    movies_r = movies_r[movies_r["rating"].isin(top_ratings)]
    if not movies_r.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        order = movies_r.groupby("rating")["duration_min"].median().sort_values(ascending=False).index
        bp = ax.boxplot(
            [movies_r[movies_r["rating"] == r]["duration_min"].values for r in order],
            patch_artist=True, notch=False,
            medianprops={"color": ACCENT_GOLD, "linewidth": 2},
            whiskerprops={"color": "#555"},
            capprops={"color": "#555"},
            flierprops={"marker": "o", "markerfacecolor": "#555", "markersize": 3, "alpha": 0.3},
        )
        for patch, color in zip(bp["boxes"], PALETTE_MAIN[:len(order)]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_xticklabels(order)
        ax.set_ylabel("Duration (minutes)")
        ax.set_title("Movie Duration Distribution by Rating")
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════
# TAB 5  –  HEATMAP
# ══════════════════════════════════════════════════════
with tab5:
    st.markdown("## Genre × Rating Heatmap")

    df_exp = df.assign(genre=df["listed_in"].str.split(", ")).explode("genre")
    top_genres  = df_exp["genre"].value_counts().head(12).index
    top_ratings = df_exp["rating"].value_counts().head(10).index
    hm_data = df_exp[df_exp["genre"].isin(top_genres) & df_exp["rating"].isin(top_ratings)]
    hm_pivot = pd.crosstab(hm_data["genre"], hm_data["rating"])

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        hm_pivot, ax=ax,
        annot=True, fmt="d",
        cmap="RdYlGn",
        linewidths=0.4, linecolor="#0d0d0d",
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Genre vs Rating — Content Count", pad=14)
    ax.set_ylabel("Genre"); ax.set_xlabel("Rating")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.markdown("## Key Insights")
    insights = [
        ("🎬", "Movies", "Movies outnumber TV Shows on Netflix — typically ~70% of the catalog."),
        ("🔞", "Ratings", "TV-MA and TV-14 are the dominant ratings, reflecting an adult-skewing content strategy."),
        ("🇺🇸", "Country", "The United States produces the highest volume of titles by a large margin."),
        ("📈", "Growth", "Netflix content additions surged sharply after 2015 and peaked around 2019–2020."),
        ("🎭", "Genres", "Drama, Comedy, and International TV Shows are the top three genres on the platform."),
        ("⏱️", "Duration", "Most movies fall in the 80–120 minute range; most TV shows run only 1–2 seasons."),
        ("👨‍🎬", "Directors", "A small group of directors account for a disproportionately large share of titles."),
        ("🔥", "Heatmap", "Children & Family content skews toward G/PG; Horror and Thriller lean heavily TV-MA."),
    ]
    cols = st.columns(2)
    for i, (icon, title, body) in enumerate(insights):
        with cols[i % 2]:
            st.markdown(
                f"""<div style="background:#1a1a1a;border-left:3px solid #e50914;
                    padding:12px 16px;border-radius:6px;margin-bottom:12px;">
                    <span style="font-size:1.3rem">{icon}</span>
                    <strong style="color:#fff;margin-left:8px">{title}</strong>
                    <p style="color:#aaa;margin:4px 0 0 0;font-size:0.88rem">{body}</p>
                </div>""",
                unsafe_allow_html=True,
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#444;font-size:0.8rem'>Netflix EDA Dashboard · Built with Streamlit</p>",
    unsafe_allow_html=True,
)