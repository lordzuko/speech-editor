import streamlit as st
from daft_exprt.synthesize import get_model, get_vocoder
from utils.session import get_state, init_session_state
from config import config, hparams

@st.cache_resource(show_spinner="Loading Daft Model...")
def load_model():
    model = get_model(config["daft_chkpt_path"], hparams)
    return model

@st.cache_resource(show_spinner="Loading Vocoder...")
def load_vocoder():
    vocoder = get_vocoder(config["vocoder_config_path"], config["vocoder_chkpt_path"])
    return vocoder