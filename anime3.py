import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import os

# =====================================
# CONFIG
# =====================================
st.set_page_config(
    page_title="Anime Dashboard",
    layout="wide"
)

st.title("🎌 Anime Analytics Dashboard")
st.markdown("Dashboard Analisis Anime 2023")

# =====================================
# DOWNLOAD USERS DETAILS
# =====================================
if not os.path.exists("users-details-2023.csv"):

    url = "https://drive.google.com/uc?id=1XQ_m3aZ34ogv5CjOA3UFLPHJ9S_RtQvc"

    gdown.download(
        url,
        "users-details-2023.csv",
        quiet=False
    )

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():

    df_anime = pd.read_csv(
        'anime-dataset-2023-clean.csv'
    )

    df_user = pd.read_csv(
        'users-details-2023.csv'
    )

    df_score = pd.read_csv(
        'users-score-small.csv'
    )

    return df_anime, df_user, df_score


df_anime, df_user, df_score = load_data()

# =====================================
# CLEAN SCORE
# =====================================
scores = df_anime[
    df_anime['Score'] != 'UNKNOWN'
]['Score']

scores = scores.astype('float')

score_mean = round(scores.mean(), 2)

df_anime['Score'] = df_anime[
    'Score'
].replace(
    'UNKNOWN',
    score_mean
)

df_anime['Score'] = df_anime[
    'Score'
].astype('float64')

# =====================================
# COLLABORATIVE FILTERING
# =====================================

rating_data = df_score[
    ['user_id', 'anime_id', 'rating']
]

anime_pivot = rating_data.pivot_table(
    index='anime_id',
    columns='user_id',
    values='rating'
).fillna(0)

anime_similarity = cosine_similarity(
    anime_pivot
)

similarity_df = pd.DataFrame(
    anime_similarity,
    index=anime_pivot.index,
    columns=anime_pivot.index
)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("Menu")

menu = st.sidebar.selectbox(
    "Pilih Halaman",
    [
        "Overview",
        "Anime Analysis",
        "User Analysis",
        "Recommendation System"
    ]
)

# =====================================
# OVERVIEW
# =====================================
if menu == "Overview":

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Anime",
        len(df_anime)
    )

    col2.metric(
        "Jumlah User",
        len(df_user)
    )

    col3.metric(
        "Jumlah Rating",
        len(df_score)
    )

    st.write("### Dataset Anime")

    st.dataframe(df_anime.head())

# =====================================
# ANIME ANALYSIS
# =====================================
elif menu == "Anime Analysis":

    st.subheader("Anime Analysis")

    # TYPE DISTRIBUTION
    st.write("## Distribusi Tipe Anime")

    type_counts = df_anime[
        'Type'
    ].value_counts()

    fig = px.bar(
        type_counts,
        x=type_counts.index,
        y=type_counts.values,
        color=type_counts.index,
        title='Count of Anime Titles by Type'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # TOP POPULAR
    st.write("## Top 15 Most Popular Anime")

    df_valid_popularity = df_anime[
        df_anime['Popularity'] > 0
    ]

    top_10_popular = df_valid_popularity.sort_values(
        by='Popularity',
        ascending=True
    ).head(15)

    fig = px.bar(
        top_10_popular,
        x='Name',
        y='Popularity',
        color='Name',
        title='Top 15 Most Popular Anime'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # SCORE VS MEMBERS
    st.write("## Score vs Members")

    fig = px.scatter(
        df_anime,
        x='Score',
        y='Members',
        color='Type',
        title='Anime Score vs Members'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # GENRE DISTRIBUTION
    st.write("## Genre Distribution")

    genre_counts = df_anime[
        df_anime['Genres'] != 'UNKNOWN'
    ]['Genres'].apply(
        lambda x: x.split(', ')
    ).explode().value_counts()

    top_20_genres = genre_counts.head(20)

    fig = px.bar(
        top_20_genres,
        x=top_20_genres.index,
        y=top_20_genres.values,
        color=top_20_genres.index,
        title='Top 20 Genres'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # WORDCLOUD
    st.write("## Genre WordCloud")

    genre_text = ' '.join(
        df_anime[
            df_anime['Genres'] != 'UNKNOWN'
        ]['Genres'].dropna()
    )

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(genre_text)

    fig_wc, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.imshow(wordcloud)

    ax.axis('off')

    st.pyplot(fig_wc)

# =====================================
# USER ANALYSIS
# =====================================
elif menu == "User Analysis":

    st.subheader("User Analysis")

    # GENDER
    gender_counts = df_user[
        'Gender'
    ].value_counts(dropna=True)

    fig = px.pie(
        values=gender_counts.values,
        names=gender_counts.index,
        title='Gender Distribution'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # LOCATION
    st.write("## Top 20 User Locations")

    location_counts = df_user[
        'Location'
    ].value_counts().head(20)

    fig = px.bar(
        location_counts,
        x=location_counts.index,
        y=location_counts.values,
        color=location_counts.index
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # AGE
    st.write("## Age Distribution")

    def calculate_age(birth_date):

        if birth_date != 'NaN':

            try:

                birth_year = int(
                    birth_date.split('-')[0]
                )

                today_year = datetime.utcnow().year

                age = today_year - birth_year

                if age >= 10 and age < 60:

                    return age

            except:

                return None

        return None

    Age = df_user[
        'Birthday'
    ].dropna().apply(calculate_age)

    fig = px.histogram(
        Age,
        nbins=20,
        title='Age Distribution'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# RECOMMENDATION SYSTEM
# =====================================
elif menu == "Recommendation System":

    st.subheader(
        "🎯 Collaborative Filtering Recommendation"
    )

    anime_list = sorted(

        df_anime[
            (df_anime['Score'] > 0)
        ]['Name']

        .dropna()

        .unique()
    )

    selected_anime = st.selectbox(
        "Pilih Anime",
        ["-- Pilih Anime --"] + anime_list
    )

    if selected_anime != "-- Pilih Anime --":

        st.write("Anime yang dipilih:")

        st.success(selected_anime)

        anime_info = df_anime[
            df_anime['Name'] == selected_anime
        ]

        # DETAIL
        st.write("### Detail Anime")

        st.dataframe(
            anime_info[
                [
                    'Name',
                    'Genres',
                    'Score',
                    'Type'
                ]
            ]
        )

        # SYNOPSIS
        st.write("### Synopsis")

        synopsis = anime_info[
            'Synopsis'
        ].values[0]

        st.info(synopsis)

        # ANIME ID
        anime_id = anime_info[
            'anime_id'
        ].values[0]

        # CEK ADA DI MATRIX
        if anime_id in similarity_df.index:

            # SIMILARITY
            similar_scores = similarity_df[
                anime_id
            ]

            # SORT
            similar_scores = similar_scores.sort_values(
                ascending=False
            )

            # TOP 10
            top_anime_ids = similar_scores.iloc[
                1:11
            ].index

            # RECOMMENDATION
            recommendations = df_anime[
                df_anime['anime_id'].isin(
                    top_anime_ids
                )
            ][
                [
                    'Name',
                    'Genres',
                    'Score'
                ]
            ]

            st.write(
                "## Anime Recommendation"
            )

            st.dataframe(
                recommendations
            )

        else:

            st.warning(
                "Anime tidak ditemukan dalam similarity matrix"
            )
