import torch
import yaml


import streamlit as st
from pages.template.login import login_screen
from utils.session import init_session_state, get_state
from config import DEBUG
from utils.db import db_init

from pages.template.se_landing import se_ui

st.set_page_config(
    page_title="Speech Editor",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="auto",
)

def main():
    if not get_state(st, "MONGO_CONNECTION"):
        st.session_state["MONGO_CONNECTION"] = db_init()
        try:
            print(get_state(st, "MONGO_CONNECTION").server_info())  # Forces a call.
        except Exception:
            raise Exception("mongo server is down.")

    # if DEBUG:
    #     se_ui()
    # else:
    login_screen()


if __name__ == "__main__":
    main()