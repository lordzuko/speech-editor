import streamlit as st
import matplotlib.pyplot as plt


from .data import process_unedited, process_edited

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
            wavdata = process_unedited()
            
            st.session_state["app"]["unedited"]["wav"] = wavdata
            st.markdown("Original:")
            st.audio(st.session_state["app"]["unedited"]["wav"],
                        sample_rate=st.session_state["sampling_rate"])
    else:
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
