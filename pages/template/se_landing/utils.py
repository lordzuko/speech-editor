import streamlit as st
from utils.audio import save_audio

def save(filename):
    """
    Save the synthesized utterances to disk and perform respective db operations
    """
    save_audio(st.session_state["app"]["unedited"]["wav"], 
               st.session_state["sampling_rate"], 
               f"data/unedited/{filename}.wav")
    save_audio(st.session_state["app"]["edited"]["wav"], 
               st.session_state["sampling_rate"], 
               f"data/edited/{filename}.wav")
    
def reset():
    """
    Reset application for new synthesis and editing job
    """
    st.session_state["app"] = {} # this will be used to track the speech data
    st.session_state["app"]["edit_next"] = True
    st.session_state["app"]["begin_processing"] = False
    st.experimental_rerun()
