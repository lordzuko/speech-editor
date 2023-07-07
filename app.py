import torch
import yaml

import certifi
from mongoengine import connect
from pymongo import ReadPreference

import streamlit as st
from pages.template.login import login_screen
from utils.session import init_session_state, get_state
from config import DB, DB_HOST, USERNAME, PASSWORD

from pages.template.se_landing import se_ui

st.set_page_config(
    page_title="Speech Editor",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="auto",
)

# if not get_state(st, "MONGO_CONNECTION"):
#     init_session_state(st, "MONGO_CONNECTION", connect(
#         host=f"mongodb+srv://{DB_HOST}/{DB}?retryWrites=true&w=majority&ssl=true",
#         # host=f"mongodb://{APP_HOST}/{APP_DB}",
#         username=USERNAME,
#         password=PASSWORD,
#         authentication_source="admin",
#         read_preference=ReadPreference.PRIMARY_PREFERRED,
#         # maxpoolsize=MONGODB_POOL_SIZE,
#         tlsCAFile=certifi.where(),
#     ))

#     try:

#         print(get_state(st, "MONGO_CONNECTION").server_info())  # Forces a call.
#     except Exception:
#         raise Exception("mongo server is down.")


def main():
    # login_screen()
    se_ui()


if __name__ == "__main__":
    main()