import streamlit as st
from config import *
from pages.ui import ui
from fs import get_model, load_model

st.set_page_config(
    page_title="Speech Editor",
    page_icon="assets/images/icon.png",
    layout="wide",
    initial_sidebar_state="auto",
)



def main():
    model = get_model()
    load_model(model)
    ui()


if __name__ == "__main__":
    main()