import streamlit as st
import numpy as np
from mongoengine.queryset.visitor import Q

from fs2.controlled_synthesis import preprocess_english
from config import lexicon, g2p, preprocess_config
from config import preprocess_config


from utils.models import Text, Annotation
from .data import setup_data
from .slider import setup_sliders
from .edits import setup_speech_unedited
from ..utils import reset, save

def se_edit_sequence():
    """
    Handler function to setup the speech editing UI
    """

    
    try:
        if st.button("Made a mistake reset?"):
            reset()

        st.session_state["app"]["data"]["t"] = dict(Text.objects(Q(wav_name__nin = st.session_state["app"]["processed_wav"]) &
                                                                Q(utt_len__lte=8))[0].to_mongo())
        
        st.session_state["app"]["text"] = st.session_state["app"]["data"]["t"]["text"]
        st.session_state["app"]["edit_next"] = False
        # st.session_state["app"]["begin_processing"] = True

        
        # if st.session_state["app"]["begin_processing"]:
        text = st.session_state["app"]["text"]

        if not st.session_state["app"]["edit_next"]:
            out = preprocess_english(text,lexicon, g2p, preprocess_config)
            texts, words, idxs = np.array([out[0]]), out[1], out[2]
            print("TEXTS: ",texts)
            setup_data(texts, words, idxs)
            st.markdown(f"Text: {st.session_state['app']['text']}")
            st.markdown(f"Filename: {st.session_state['app']['data']['t']['wav_name']}")
            
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
                with st.form("Save"):
                    st.markdown("Finalize editing?")
                    submitted = st.form_submit_button("Complete")
                    if submitted:
                        st.info("Saving to DB")
                        save(st.session_state['app']['data']['t']['wav_name'])
                        st.success("Saved!")
                        st.session_state["app"]["processed_wav"].append(st.session_state["app"]["data"]["t"]["wav_name"])
                        reset()
                
    except IndexError:
        st.success("No new items in DB")
                                    

def se_edit_single():
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
                        save(st.session_state['app']['filename'])
                        reset()
                       

    return words_to_edit
