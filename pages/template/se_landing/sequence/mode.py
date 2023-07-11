import streamlit as st
import numpy as np
from mongoengine.queryset.visitor import Q

from fs2.controlled_synthesis import preprocess_english
from config import lexicon, g2p, preprocess_config
from config import preprocess_config


from utils.models import Text
from utils.db import handle_submit
from .data import setup_data
from .slider import setup_sliders
from .edits import setup_speech_unedited
from ..utils import reset_sequence, save

def se_edit_sequence():
    """
    Handler function to setup the speech editing UI
    """

    
    try:
        if st.button("Made a mistake reset?"):
            reset_sequence()

        st.session_state["app"]["data"]["t"] = dict(Text.objects(Q(wav_name__nin = st.session_state["processed_wav"]) &
                                                                Q(utt_len__lte=8))[0].to_mongo())
        
        st.session_state["app"]["text"] = st.session_state["app"]["data"]["t"]["text"]
        st.session_state["app"]["wav_name"] = st.session_state['app']['data']['t']['wav_name']
        st.session_state["app"]["edit_next"] = False
        text = st.session_state["app"]["text"]

        if not st.session_state["app"]["edit_next"]:
            out = preprocess_english(text,lexicon, g2p, preprocess_config)
            texts, words, idxs = np.array([out[0]]), out[1], out[2]
            print("TEXTS: ",texts)
            setup_data(texts, words, idxs)
            st.markdown(f"#### Text: {st.session_state['app']['text']}")
            st.markdown(f"##### Filename: {st.session_state['app']['data']['t']['wav_name']}")
            
            if not st.session_state["app"].get("suggestions"):
                suggestions = [f"{w}-{i}" for i,w in enumerate(st.session_state["app"]["w"])]
                st.session_state["app"]["suggestions"] = suggestions
            
            col1, col2 = st.columns([2, 2])
            with col1:
                setup_speech_unedited()

            if st.session_state["app"]["suggestions"]:
                if "unedited" in st.session_state["app"]:
                    if "wav" in st.session_state["app"]["unedited"]:
                        setup_sliders(column=col2)

            
            st.markdown("---")
            c1, _ = st.columns([1, 1])
            
            with c1:
                next_bt = st.button("Next")
                if next_bt:
                    st.info("Saving to DB")
                    save(st.session_state["app"]["wav_name"], username=st.session_state["login"]["username"])
                    print("save value: ", handle_submit())
                    st.success("Saved!")
                    reset_sequence()

    except IndexError:
        st.success("No new items in DB")
                                    
