import datetime
import torch
from operator import itemgetter
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.audio import save_audio
from utils.session import get_state, init_session_state
from fs2.controlled_synthesis import  preprocess_single, synthesize, preprocess_english
from config import lexicon, g2p, args, preprocess_config, configs, STATS

# from st_row_buttons import st_row_buttons
# from st_btn_select import st_btn_select
from fs2.utils.model import get_model, get_vocoder
from config import args, configs, device, model_config, preprocess_config, DEBUG

from utils.models import SEData
from utils.data import setup_data, process_unedited, process_edited

from streamlit_tags import st_tags

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

def setup_sliders(column):
    """
    Handle sliders on UI
    """
    print(st.session_state["app"]["suggestions"])
    for word in st.session_state["app"]["suggestions"]:
        # word = st.session_state["app"]["current_word"]
        w = word.split('-')[0]
        idx = int(word.split("-")[1])

        with st.form(key=f"form-{word}"):

            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            with col2:
                
                duration_control = st.slider("Duration Scale", 
                    value=float(st.session_state["app"]["fc"]["word"]["d"][0][idx]), 
                    min_value=float(max(0, -3*STATS["gs"]["d"]["std"] + STATS["gs"]["d"]["mean"])), 
                    max_value=float(3*STATS["gs"]["d"]["std"] + STATS["gs"]["d"]["mean"]), 
                    help="Speech speed. Larger value become slow",
                    # step=1,
                    key=f"duration-{word}")
                
            with col3:
                f0_control = st.slider("Pitch Scale", 
                    value=float(st.session_state["app"]["fc"]["word"]["p"][0][idx]), 
                    min_value=float(max(0, -3*STATS["gs"]["p"]["std"] + STATS["gs"]["p"]["mean"])), 
                    max_value=float(3*STATS["gs"]["p"]["std"] + STATS["gs"]["p"]["mean"]),
                    # step=1,
                    key=f"pitch-{word}")
                

            with col4:
                energy_control = st.slider("Energy Scale", 
                    value=float(st.session_state["app"]["fc"]["word"]["p"][0][idx]), 
                    min_value=float(max(0, -3*STATS["gs"]["e"]["std"] + STATS["gs"]["e"]["mean"])), 
                    max_value=float(3*STATS["gs"]["e"]["std"] + STATS["gs"]["e"]["mean"]),
                    # step=1,
                    key=f"energy-{word}")

                # st.markdown(f"Currently Editing: `{word}` at index `{cur_word_idx}`")            
            
            with col1:
                submitted = st.form_submit_button(f"{word}")
                
                if submitted:
                    st.markdown(duration_control)
                    st.markdown(f0_control)
                    st.markdown(energy_control)

                    st.session_state["app"]["fc"]["word"]["d"][0][idx] = duration_control
                    st.session_state["app"]["fc"]["word"]["p"][0][idx] = f0_control
                    st.session_state["app"]["fc"]["word"]["e"][0][idx] = energy_control
                    
                    with column:
                        # if DEBUG:
                        #     st.json(st.session_state["app"])
                        setup_speech_edited()
                        with st.expander("Spectrogram visualization"):
                            fig = plt.figure()
                            ax1 = fig.add_subplot(1, 1, 1)
                            ax1.specgram(st.session_state["app"]["edited"]["wav"],
                                            Fs=st.session_state["sampling_rate"])
                            st.pyplot(fig)



        
def setup_speech_unedited():
    """
    Handle unedited speech which works as reference
    """
    if "unedited" not in st.session_state["app"]:
        st.session_state["app"]["unedited"] = {}
        st.session_state["app"]["unedited"]["synthesized"] = False
    
    if not st.session_state["app"]["unedited"]["synthesized"]:
        st.session_state["app"]["unedited"]["synthesized"] =  st.button("Synth!", 
                                        disabled=st.session_state["app"]["unedited"]["synthesized"])
        
        if st.session_state["app"]["unedited"]["synthesized"]:
            print(",asdf", st.session_state["app"]["unedited"])
            wavdata = process_unedited()
            print(st.session_state["app"]["fc"])
            
            st.session_state["app"]["unedited"]["wav"] = wavdata
            st.markdown("Original:")
            st.audio(st.session_state["app"]["unedited"]["wav"],
                        sample_rate=st.session_state["sampling_rate"])
    else:
        print("there:", st.session_state["app"]["unedited"])
        st.markdown("Original:")
        st.audio(st.session_state["app"]["unedited"]["wav"],
                            sample_rate=st.session_state["sampling_rate"])
        
        with st.expander("Spectrogram visualization"):
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.specgram(st.session_state["app"]["unedited"]["wav"],
                            Fs=st.session_state["sampling_rate"])
            st.pyplot(fig)



def setup_speech_edited():
    """
    Handle edited speech which annotator can use to
    check the changes the edits are making
    """
    if "edited" not in st.session_state["app"]:
        st.session_state["app"]["edited"] = {}
        st.session_state["app"]["unedited"]["synthesized"] = False

    wavdata = process_edited()

    st.markdown("Edited:")
    st.session_state["app"]["edited"]["wav"] = wavdata
    st.audio(st.session_state["app"]["edited"]["wav"],
                sample_rate=st.session_state["sampling_rate"])

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
        print("beging: ", st.session_state["app"]["begin_processing"]) 
        print("making edit_text = False")
        st.session_state["app"]["edit_next"] = False

    print("beging:-- ", st.session_state["app"]["begin_processing"]) 

    if st.session_state["app"]["begin_processing"]:
        text = st.session_state["app"]["text"]

        print("beginng!")
        if not st.session_state["app"]["edit_next"]:
            out = preprocess_english(text,lexicon, g2p, preprocess_config)
            texts, words, idxs = np.array([out[0]]), out[1], out[2]

            # st.markdown(texts)
            setup_data(texts, words, idxs)
            # st.markdown(words)
            # st.markdown(idxs)
            st.markdown(f"Text: {st.session_state['app']['text']}")
            st.markdown(f"Filename: {st.session_state['app']['filename']}")

            # if "fc" not in st.session_state["app"]:
            #     st.session_state["app"]["fc"] = {
            #         "f0": np.ones((1,len(words))),
            #         "energy": np.ones((1,len(words))),
            #         "duration": np.ones((1,len(words)))
            #     }
            # setup_data
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

            if not st.session_state["app"]["suggestions"] and "edited" in st.session_state["app"]:

                with col2:
                     st.markdown("Edited:")
                     st.audio(st.session_state["app"]["edited"]["wav"],
                        sample_rate=st.session_state["sampling_rate"])

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
        print(dict(st.session_state).keys())

        if (not get_state(st, "model")) or (not get_state(st, "vocoder")) or (not get_state(st, "sampling_rate")):
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
