import datetime
import re
import streamlit as st
import certifi
from mongoengine import connect
from pymongo import ReadPreference

from utils import check_hashes, make_hashes
from utils.models import Users
from utils.text import get_random_string
from config import DB, DB_HOST, USERNAME, PASSWORD, DEBUG


def validate_login(username, password):
    fetched_data = Users.objects(username=username)
    
    if fetched_data:
        fetched_data = fetched_data[0]
        
        if check_hashes(password, fetched_data["password"]):
            return dict(fetched_data.to_mongo())

    return {}


def update_password(username, new_password=""):

    fetched_data = Users.objects(username=username)
    if fetched_data:
        fetched_data = fetched_data[0]

        if not new_password:
            new_password = get_random_string(10)
            print("Random New Password:", new_password)

        success = fetched_data.update(
            set__password=make_hashes(new_password),
            set__modified_at=datetime.datetime.utcnow(),
        )

        # TODO: check for success
        # Trigger email
        return True

    return False


def delete_account(username):
    fetched_data = Users.objects(username=username)
    if fetched_data:
        fetched_data = fetched_data[0]

        success = fetched_data.update(
            set__deactivated=True, set__modified_at=datetime.datetime.utcnow()
        )
        return True
    return False


def fetch_annotators(user_type):
    fetched_data = Users.objects(user_type=user_type).only("username")
    data = []
    if fetched_data:
        for d in fetched_data:
            d = dict(d.to_mongo())
            d.pop("_id")
            data.append(d["username"])
    print(data)
    return data

@st.cache_resource(show_spinner="Connecting to DB")
def db_init():
    return connect(
        host=f"mongodb+srv://{DB_HOST}/{DB}?retryWrites=true&w=majority&ssl=true",
        # host=f"mongodb://{APP_HOST}/{APP_DB}",
        username=USERNAME,
        password=PASSWORD,
        authentication_source="admin",
        read_preference=ReadPreference.PRIMARY_PREFERRED,
        # maxpoolsize=MONGODB_POOL_SIZE,
        tlsCAFile=certifi.where(),
    )
