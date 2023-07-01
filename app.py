import torch
import yaml
import streamlit as st
from ui import ui
from utils.session import init_session_state, get_state
from fs2.utils.model import get_model, get_vocoder
from config import args, configs, device, model_config, preprocess_config

st.set_page_config(
    page_title="Speech Editor",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="auto",
)

def main():


    with st.spinner("Loading and setting up TTS model..."):
        if not get_state(st, "model"):
            # Get model
            print("Loading Model...")
            init_session_state(st, "model", get_model(args, configs, device, train=False))
            print("Model Loaded")
        
        if not get_state(st, "vocoder"):
            # Load vocoder
            print("Loading Vocoder...")
            init_session_state(st, "vocoder", get_vocoder(model_config, device))
            print("Vocoder Loaded")

        if not get_state(st, "sampling_rate"):
            init_session_state(st, "sampling_rate", preprocess_config["preprocessing"]["audio"]["sampling_rate"])
    
    ui()


if __name__ == "__main__":
    main()