import streamlit as st
from config import *
from pages.ui import ui

st.set_page_config(
    page_title="Speech Editor",
    page_icon="assets/images/icon.png",
    layout="wide",
    initial_sidebar_state="auto",
)

def main():
    ui()


if __name__ == "__main__":
    main()