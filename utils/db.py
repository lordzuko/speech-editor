import datetime
import re
import streamlit as st
import certifi
from mongoengine import connect
from pymongo import ReadPreference
import numpy as np
from collections import defaultdict

from utils import check_hashes, make_hashes
from utils.models import Users, Annotation
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

def fetch_annotated(tagger):
    fetched_data = Annotation.objects(tagger=tagger).only("wav_name")
    data = []
    if fetched_data:
        for d in fetched_data:
            d = dict(d.to_mongo())
            data.append(d["wav_name"])
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

def _data2mongo_(data: dict):
    """
    convert the numpy values to array/ list type 
    to make them compatible with mongodb's accepted 
    datatypes
    """
    out = defaultdict(dict)
    keys = ["phone", "scaling", "word", "mask", "gl_d", "gl_p", "gl_e"]
    for k, v in data.items():
        if k in keys:
            if isinstance(v, int) or isinstance(v, float):
                out[k] = v
                continue
            else:
                for kk, vv in v.items():
                    if isinstance(vv, np.ndarray):
                        out[k][kk] = vv.tolist()
                    else:
                        out[k][kk] = vv
    return out

def handle_submit():
    annot = dict()
    annot["text"] = st.session_state["app"]["text"]
    annot["unedited"] = _data2mongo_(st.session_state["app"]["unedited"])
    if "edited" in st.session_state["app"]:
        annot["edited"] = _data2mongo_(st.session_state["app"]["fc"])
        print(annot["edited"].keys())

    annot["wav_name"] = st.session_state["app"]["wav_name"]
    annot["p"] = st.session_state["app"]["p"]
    annot["w"] = st.session_state["app"]["w"]
    annot["idxs"] = st.session_state["app"]["idxs"]
    annot["i2p"] = {str(k): v for k, v in st.session_state["app"]["i2p"].items()}
    annot["i2w"] = {str(k): v for k,v in st.session_state["app"]["i2w"].items()}
    annot["w2p"] = {str(k) : v for k,v in st.session_state["app"]["w2p"].items()}
    annot["created_at"] = datetime.datetime.utcnow()
    annot["tagger"] = st.session_state["login"]["username"]
    annot["tagged_at"] = datetime.datetime.utcnow()
    print(annot)
    success = Annotation(**annot).save(check_keys=False)
    st.session_state["processed_wav"].append(st.session_state["app"]["wav_name"])
    return success
