import os
import uuid
import streamlit as st
import matplotlib.pyplot as plt
import librosa
from config import ref_style, unedited_path
from .data import process_unedited, process_edited
from daft_exprt.extract_features import rescale_wav_to_float32
from scipy.io import wavfile


def setup_ref_speech():
    st.markdown("Reference Audio:")
    style = st.session_state["app"]["data"]["t"]["ref_style"]
    wav, fs = librosa.load(ref_style[style], sr=st.session_state["sampling_rate"])
    wav = rescale_wav_to_float32(wav)
    st.audio(wav, sample_rate=st.session_state["sampling_rate"])

def setup_speech_unedited():
    """
    Handle unedited speech which works as reference
    """
    if "unedited" not in st.session_state["app"]:
        st.session_state["app"]["unedited"] = {}
        wavdata = process_unedited()
        
        st.session_state["app"]["unedited"]["wav"] = wavdata
        st.markdown("Original:")
        st.audio(st.session_state["app"]["unedited"]["wav"],
                    sample_rate=st.session_state["sampling_rate"])
        # SAVE FILE
        filename = st.session_state["app"]["save_wav_name"]
        wavfile.write(f"{os.path.join(unedited_path, filename)}", st.session_state["sampling_rate"], wavdata)
        
    else:
        st.markdown("Original:")
        st.audio(st.session_state["app"]["unedited"]["wav"],
                            sample_rate=st.session_state["sampling_rate"])
        
        # with st.expander("Spectrogram visualization"):
        #     fig = plt.figure()
        #     ax1 = fig.add_subplot(1, 1, 1)
        #     ax1.specgram(st.session_state["app"]["unedited"]["wav"],
        #                     Fs=st.session_state["sampling_rate"])
        #     st.pyplot(fig)


def setup_speech_edited():
    """
    Handle edited speech which annotator can use to
    check the changes the edits are making
    """
    if "edited" not in st.session_state["app"]:
        st.session_state["app"]["edited"] = {}
    wavdata = process_edited()

    st.markdown("Edited:")
    st.session_state["app"]["edited"]["wav"] = wavdata
    st.audio(st.session_state["app"]["edited"]["wav"],
                sample_rate=st.session_state["sampling_rate"])
