import datetime
import streamlit as st

from utils.models import Annotation
from utils.session import get_state, init_session_state

from config import preprocess_config
from config import args as _args
from config import configs as _configs
from config import device as _device
from config import model_config as _model_config

from fs2.utils.model import get_model, get_vocoder
from config import preprocess_config,  MODE

from .mode import se_edit_single, se_edit_sequence
from .utils import reset

def handle_submit(data, stage="final"):
    
    annot = dict()
    if stage == "final": 
        annot["text"] = data["text"]
        annot["unedited"] = data["unedited"]
        annot["edited"] = data["edited"]
        annot["wav_name"] = data["wav_name"]
        output.created_at = datetime.datetime.utcnow()
        output.tagger = tagger
        output.tagged_at = datetime.datetime.utcnow()
        output.save()
        success = Annotation(**data).save()
        #, set__tagger=tagger, set__tagged_at=datetime.datetime.utcnow()
        data.update(set__tagging_status = 'tagged')
        data.save()
        st.session_state["processed_essay_ids"].append(dict(data.to_mongo())['userId'])
        print('Your submitted response')

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
                    init_session_state(st, "model", get_model(_args, _configs, _device, _train=False))
                    print("Model Loaded")
                
                if not get_state(st, "vocoder"):
                    # Load vocoder
                    print("Loading Vocoder...")
                    init_session_state(st, "vocoder", get_vocoder(_model_config, _device))
                    print("Vocoder Loaded")

                if not get_state(st, "sampling_rate"):
                    init_session_state(st, "sampling_rate", preprocess_config["preprocessing"]["audio"]["sampling_rate"])
    
        st.header("Speech Editor")
        if MODE == "single":
            se_edit_single()
        else:
            st.session_state["app"]["processed_wav"] = []
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