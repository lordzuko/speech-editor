import streamlit as st
import base64

from utils.audio import save_audio
from streamlit.components.v1 import html
from pathlib import Path
# def autoplay_audio():
#     print("autoplayig!!!")
#     b64 = base64.b64encode(st.session_state["app"]["edited"]["wav"]).decode('utf-8')
#     audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{b64}">'
#     st.write(audio_tag, unsafe_allow_html=True)

def autoplay_audio():
    js = """
    <script>
        var audio = window.parent.document.querySelectorAll('.stAudio')[2]

        var editButtons = window.parent.document.querySelectorAll('.e1ewe7hr5');
    
        editButtons.forEach((el => el.addEventListener('click', function(){
            setTimeout(function () {
                audio.play();
            }, 2000);
        }, true)));
        
        var uttLevelButton = window.parent.document.querySelectorAll('.e1ewe7hr5')[1];

        window.parent.document.onkeyup = function(e) {
            if (e.which == 13) {
                console.log('trigger');
                uttLevelButton.click();
            }};
        
    </script>
    """
    # st.markdown(js, unsafe_allow_html=True)
    # ret_val = st_javascript(js)
    # print("JS: RETURN:", ret_val)
    # st.markdown(js, unsafe_allow_html=True)
    html(js, height=1)

def save():
    """
    Save the synthesized utterances to disk and perform respective db operations
    """
    # filename = st.session_state["app"]["save_wav_name"]
    # wavdata = st.session_state["app"]["edited"]["wav"]
    # wavfile.write(f"{os.path.join(edited_path, filename)}", st.session_state["sampling_rate"], wavdata)
    username = st.session_state["login"]["username"]
    unedited = f"data/{username}/unedited/"
    edited = f"data/{username}/edited/"

    Path(unedited).mkdir(parents=True, exist_ok=True)
    Path(edited).mkdir(parents=True, exist_ok=True)

    save_audio(st.session_state["app"]["unedited"]["wav"], 
               st.session_state["sampling_rate"], 
               f"{unedited}/{st.session_state['app']['save_wav_name']}")
    
    if "edited" in st.session_state["app"]:
        save_audio(st.session_state["app"]["edited"]["wav"], 
                st.session_state["sampling_rate"], 
                f"{edited}/{st.session_state['app']['save_wav_name']}")
    
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


