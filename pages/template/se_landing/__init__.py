import streamlit as st

from utils.session import get_state, init_session_state
from utils.db import fetch_annotated
from utils.tts_model import load_model, load_vocoder

from config import  MODE, hparams

from .sequence.mode import se_edit_sequence
from .single.mode import se_edit_single
from .utils import reset

def se_ui():        
    """
    Init Speech Editing UI page
    """
    if not st.session_state.get("app"):
        st.session_state["app"] = {} # this will be used to track the speech data

    if st.session_state.get("is_tagging_started"):
        # print(dict(st.session_state).keys())

        if (not get_state(st, "model")) or (not get_state(st, "vocoder")) or (not get_state(st, "sampling_rate")):
            with st.spinner("Loading and setting up TTS model..."):
                if not get_state(st, "model"):
                    # Get model
                    print("Loading Model...")
                    # init_session_state(st, "model", get_model(_args, _configs, _device, _train=False))
                    init_session_state(st, "model", load_model())
                    print("Model Loaded")
                
                if not get_state(st, "vocoder"):
                    # Load vocoder
                    print("Loading Vocoder...")
                    init_session_state(st, "vocoder", load_vocoder())
                    print("Vocoder Loaded")

                if not get_state(st, "sampling_rate"):
                    init_session_state(st, "sampling_rate", hparams.sampling_rate)
    
        st.header("Speech Editor")
        if MODE == "single":
            se_edit_single()
        else:
            if not st.session_state.get("processed_wav"):
                st.session_state["processed_wav"] = fetch_annotated(st.session_state["login"]["username"])
            if not st.session_state["app"].get("data"):
                st.session_state["app"]["data"] = {}
            se_edit_sequence()
    else:
        st.subheader("Tagging Notes")
        st.markdown("""
            * <Add Notes for taggers here>
            * Once the correct options have been selected, click, `Start Tagging` to proceed.
        """)
        start_bt = st.button("Click here to start tagging")
        if start_bt:
            st.session_state["is_tagging_started"] = True
            reset()