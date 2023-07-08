import streamlit as st
import matplotlib.pyplot as plt
from .data import sample_gauss, standardize
from .edits import setup_speech_edited
from config import DEBUG, STATS

# def setup_bias_slider(column):

#     with col2:
#         duration_control = st.slider("Global Duration",
#                                         value=0.,
#                                         min_value=-100.,
#                                         max_value=100.,
#                                         step=1,
#                                         key=f"duration-global") 


def setup_sliders(column):
    """
    Handle sliders on UI
    """
    print("Suggestions: ", st.session_state["app"]["suggestions"])
    bias = False
    for word in st.session_state["app"]["suggestions"]:
        # word = st.session_state["app"]["current_word"]
        w = word.split('-')[0]
        idx = int(word.split("-")[1])

        with st.form(key=f"form-{word}"):

            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

            with col2:
                
                duration_control = st.slider("Duration Scale", 
                    value=int(st.session_state["app"]["fc"]["word"]["d"][0][idx]), 
                    min_value=0,  
                    max_value=int(sample_gauss(3, STATS["gs"]["d"]["mean"],STATS["gs"]["d"]["std"])), 
                    help="Speech speed. Larger value become slow",
                    step=1,
                    key=f"duration-{word}")
                
            with col3:
                p = st.session_state["app"]["fc"]["word"]["p"][0][idx]
                f0_control = st.slider("Pitch Scale", 
                    value=float(sample_gauss(p, STATS["gs"]["p"]["mean"],STATS["gs"]["p"]["std"])), 
                    min_value=0., 
                    max_value=float(round(sample_gauss(3, STATS["gs"]["p"]["mean"],STATS["gs"]["p"]["std"]))),
                    step=1.,
                    key=f"pitch-{word}")
                

            with col4:
                e = st.session_state["app"]["fc"]["word"]["e"][0][idx]
                energy_control = st.slider("Energy Scale", 
                    value=float(sample_gauss(e, STATS["gs"]["e"]["mean"],STATS["gs"]["e"]["std"])), 
                    min_value=0., 
                    max_value=float(round(sample_gauss(3, STATS["gs"]["e"]["mean"],STATS["gs"]["e"]["std"]))),
                    step=1.,
                    key=f"energy-{word}")

                # st.markdown(f"Currently Editing: `{word}` at index `{cur_word_idx}`")            
            
            with col1:
                submitted = st.form_submit_button(f"{word}")
                
                if submitted:
                    
                    # duration_control = standardize(duration_control, STATS["gs"]["d"]["mean"],STATS["gs"]["d"]["std"])
                    f0_control = standardize(f0_control, STATS["gs"]["p"]["mean"],STATS["gs"]["p"]["std"])
                    energy_control = standardize(energy_control, STATS["gs"]["e"]["mean"],STATS["gs"]["e"]["std"])

                    st.session_state["app"]["fc"]["word"]["d"][0][idx] = duration_control
                    st.session_state["app"]["fc"]["word"]["p"][0][idx] = f0_control
                    st.session_state["app"]["fc"]["word"]["e"][0][idx] = energy_control
                    
                    if DEBUG:
                        st.markdown(duration_control)
                        st.markdown(energy_control)
                        st.markdown(f0_control)
                    with column:
                        setup_speech_edited()
                        with st.expander("Spectrogram visualization"):
                            fig = plt.figure()
                            ax1 = fig.add_subplot(1, 1, 1)
                            ax1.specgram(st.session_state["app"]["edited"]["wav"],
                                            Fs=st.session_state["sampling_rate"])
                            st.pyplot(fig)

    if DEBUG:
        st.markdown("word2phone mapping:")
        st.json(st.session_state["app"]["w2p"])
        st.markdown("Edited:")
        st.json(st.session_state["app"]["fc"])
        st.markdown("Unedited:")
        st.json(st.session_state["app"]["unedited"])


        
