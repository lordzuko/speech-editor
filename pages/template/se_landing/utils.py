import streamlit as st
import base64

from utils.audio import save_audio

def autoplay_audio():
    print("autoplayig!!!")
    b64 = base64.b64encode(st.session_state["app"]["edited"]["wav"]).decode('utf-8')
    audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{b64}">'
    st.markdown(audio_tag, unsafe_allow_html=True)

def save():
    """
    Save the synthesized utterances to disk and perform respective db operations
    """
    # filename = st.session_state["app"]["save_wav_name"]
    # wavdata = st.session_state["app"]["edited"]["wav"]
    # wavfile.write(f"{os.path.join(edited_path, filename)}", st.session_state["sampling_rate"], wavdata)
    save_audio(st.session_state["app"]["unedited"]["wav"], 
               st.session_state["sampling_rate"], 
               f"data/unedited/{st.session_state['app']['save_wav_name']}")
    
    if "edited" in st.session_state["app"]:
        save_audio(st.session_state["app"]["edited"]["wav"], 
                st.session_state["sampling_rate"], 
                f"data/edited/{st.session_state['app']['save_wav_name']}")
    
def reset():
    """
    Reset application for new synthesis and editing job
    """
    st.session_state["app"] = {} # this will be used to track the speech data
    st.session_state["app"]["edit_next"] = True
    st.session_state["app"]["begin_processing"] = False
    st.experimental_rerun()


def reset_sequence():
    """
    Reset states for sequential tagging task
    """
    st.session_state["app"] = {} # this will be used to track the speech data
    st.session_state["app"]["edit_next"] = True
    st.session_state["app"]["data"] = {}
    st.experimental_rerun()


