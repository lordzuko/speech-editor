import datetime
import torch
from pprint import pprint
from operator import itemgetter
import streamlit as st
import numpy as np
from utils.audio import save_audio
from utils.session import get_state, init_session_state
from fs2.controlled_synthesis import  preprocess_single, synthesize, preprocess_english
from config import lexicon, g2p, preprocess_config

from config import args as _args
from config import configs as _configs
from config import device as _device
from config import model_config as _model_config

# from st_row_buttons import st_row_buttons
# from st_btn_select import st_btn_select
from streamlit_tags import st_tags
from fs2.utils.model import get_model, get_vocoder
from config import preprocess_config, DEBUG

from utils.models import SEData
from .data import setup_data
from .slider import setup_sliders
from .edits import setup_speech_unedited


def handle_submit(data , ques,ans, tagger):
    
    output = SEData(wav_file = dict(data.to_mongo())['wav_file'])
    output.essay_question = ques
    output.essay_answer = ans
    output.essay_link = dict(data.to_mongo())['essay_link']
    output.created_at = datetime.datetime.utcnow()
    output.tagger = tagger
    output.tagged_at = datetime.datetime.utcnow()
    output.save()
    #, set__tagger=tagger, set__tagged_at=datetime.datetime.utcnow()
    data.update(set__tagging_status = 'tagged')
    data.save()
    st.session_state["processed_essay_ids"].append(dict(data.to_mongo())['userId'])
    print('Your submitted response')



def tag_ui(suggestions, values):
    words_to_edit = st_tags(
        label="Add words to edit",
        text="enter word for suggestion and press enter",
        value=values,
        suggestions=suggestions)
    return words_to_edit

def save():
    """
    Save the synthesized utterances to disk and perform respective db operations
    """
    save_audio(st.session_state["app"]["unedited"]["wav"], 
               st.session_state["sampling_rate"], 
               f"data/unedited/{st.session_state['app']['filename']}.wav")
    save_audio(st.session_state["app"]["edited"]["wav"], 
               st.session_state["sampling_rate"], 
               f"data/edited/{st.session_state['app']['filename']}.wav")
    
    # TODO: perform the db operations
    
def reset():
    """
    Reset application for new synthesis and editing job
    """
    st.session_state["app"] = {} # this will be used to track the speech data
    st.session_state["app"]["edit_next"] = True
    st.session_state["app"]["begin_processing"] = False
    st.experimental_rerun()

def se_edit_widget():
    """
    Handler function to setup the speech editing UI
    """
    if st.button("Made a mistake reset?"):
        reset()

    words_to_edit = []
    
    if st.session_state["app"]["edit_next"]:
        with st.form("Enter details:"):
            st.session_state["app"]["text"] = st.text_area("Text", value="", height=100, max_chars=2048)
            st.session_state["app"]["filename"] = st.text_input("Filename")
            submitted = st.form_submit_button("Submit")
                
                
    if st.session_state["app"].get("text"):
        st.session_state["app"]["begin_processing"] = True
        st.session_state["app"]["edit_next"] = False


    if st.session_state["app"]["begin_processing"]:
        text = st.session_state["app"]["text"]

        if not st.session_state["app"]["edit_next"]:
            out = preprocess_english(text,lexicon, g2p, preprocess_config)
            texts, words, idxs = np.array([out[0]]), out[1], out[2]

            setup_data(texts, words, idxs)
            st.markdown(f"Text: {st.session_state['app']['text']}")
            st.markdown(f"Filename: {st.session_state['app']['filename']}")

            setup_data(texts, words, idxs)
            
            if not st.session_state["app"].get("suggestions"):
                suggestions = [f"{w}-{i}" for i,w in enumerate(st.session_state["app"]["w"])]
                st.session_state["app"]["suggestions"] = suggestions
            
            col1, col2 = st.columns([2, 2])
            with col1:
                setup_speech_unedited()

            if st.session_state["app"]["suggestions"]:
                if "unedited" in st.session_state["app"]:
                    if st.session_state["app"]["unedited"]["synthesized"]:
                        setup_sliders(column=col2)

            if "fc" in st.session_state["app"]:
                with st.form("Finalize editing?:"):
                    st.markdown("Finalize editing?")
                    submitted = st.form_submit_button("Complete")
                    if submitted:
                        save()
                        reset()
                       

    return words_to_edit

def se_ui():        
    """
    Init Speech Editing UI page
    """
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
        se_edit_widget()
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
