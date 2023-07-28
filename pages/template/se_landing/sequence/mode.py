import os
import uuid
import streamlit as st
import numpy as np
from datetime import datetime
from mongoengine.queryset.visitor import Q
from scipy.io import wavfile

from daft_exprt.synthesize import prepare_sentences_for_inference
from config import hparams, dictionary, edited_path

from utils.models import Text
from utils.db import handle_submit
from .data import setup_data
from .slider import setup_sliders
from .edits import setup_speech_unedited, setup_ref_speech
from ..utils import reset_sequence, save

def se_edit_sequence():
    """
    Handler function to setup the speech editing UI
    """

    
    # try:
    if st.button("Made a mistake reset?"):
        reset_sequence()

    st.session_state["app"]["data"]["t"] = dict(Text.objects(Q(wav_name__nin = st.session_state["processed_wav"]) &
                                                            Q(utt_len__lte=8))[0].to_mongo())
    
    st.session_state["app"]["text"] = st.session_state["app"]["data"]["t"]["text"]
    st.session_state["app"]["wav_name"] = st.session_state['app']['data']['t']['wav_name']
    if not st.session_state["app"].get("save_wav_name"):
        st.session_state["app"]["save_wav_name"] = str(uuid.uuid4()) + ".wav"
    st.session_state["app"]["edit_next"] = False
    text = st.session_state["app"]["text"]

    if not st.session_state["app"]["edit_next"]:
        # out = preprocess_english(text,lexicon, g2p, preprocess_config)
        if not st.session_state["app"].get("phone_sents"):
            print("PREPARING SENTENCES:")
            out = prepare_sentences_for_inference([text], dictionary, hparams)

            st.session_state["app"]["phone_sents"], words, phones, idxs, st.session_state["app"]["ignore_idxs"] = [out[0][0]], out[0][1], out[0][2], out[0][3], out[0][4]

            print("TEXTS: ",st.session_state["app"]["phone_sents"])
            setup_data(words, phones, idxs)
            st.session_state["app"]["num_edits"] = 0
            st.session_state["app"]["edit_start"] = datetime.now()

        st.markdown(f"#### Text: {st.session_state['app']['text']}")
        st.markdown(f"##### Filename: {st.session_state['app']['data']['t']['wav_name']}")
        
        if not st.session_state["app"].get("suggestions"):
            suggestions = [f"{w}-{i}" for i,w in enumerate(st.session_state["app"]["w"])]
            st.session_state["app"]["suggestions"] = suggestions
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            setup_ref_speech()
        with col2:
            setup_speech_unedited()

        if st.session_state["app"]["suggestions"]:
            if "unedited" in st.session_state["app"]:
                if "wav" in st.session_state["app"]["unedited"]:
                    setup_sliders(column=col3)

        
        st.markdown("---")
        c1, _ = st.columns([1, 1])
        
        with c1:
            next_bt = st.button("Next")
            if next_bt:
                st.info("Saving to DB")
                save()
                print("save value: ", handle_submit())
                
                st.success("Saved!")
                reset_sequence()

    # except IndexError:
    #     st.success("No new items in DB")
                                    
