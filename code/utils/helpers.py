import os
import glob
import gzip
import pandas as pd
import streamlit as st

def load_styles():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def add_footer(fig, data_source="SofaScore", prepared_by="@urazdev", extra_text=None, x=0.99, y=-0.05, fontsize=10, ha="right"):
    footer_text = f"Data: {data_source}\nPrepared by {prepared_by}"
    if extra_text:
        footer_text += f"\n{extra_text}"

    fig.text(
        x, y,
        footer_text,
        ha=ha,
        va="bottom",
        fontsize=fontsize,
        fontstyle="italic",
        color="gray"
    )

def render_spinner(content_function, *args, **kwargs):
    with st.spinner("Preparing content..."):
        content_function(*args, **kwargs)

def load_with_spinner(load_function, *args, **kwargs):
    with st.spinner("Loading players..."):
        return load_function(*args, **kwargs)

@st.cache_data(show_spinner=False)
def load_filtered_json_files(directory: str, country: str, league: str, season: str, subdirectory: str) -> pd.DataFrame:
    path = os.path.join(directory, subdirectory, f"sofascore_{country}_{league}_{season}_{subdirectory}.json.gz")
    files = glob.glob(path)

    dataframes = []
    for file in files:
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            dataframes.append(pd.read_json(f))

    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

def turkish_sort_key():
    turkish_alphabet = "AaBbCcÇçDdEeFfGgĞğHhIıİiJjKkLlMmNnOoÖöPpRrSsŞşTtUuÜüVvYyZz"
    turkish_sort_order = {char: idx for idx, char in enumerate(turkish_alphabet)}

    def sort_function(text):
        return [turkish_sort_order.get(char, -1) for char in text]

    return sort_function

def sort_turkish(df, column):
    sort_key = turkish_sort_key()
    return df.sort_values(by=column, key=lambda col: col.map(sort_key))