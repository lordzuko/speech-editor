import torch
import streamlit as st
import matplotlib.pyplot as plt
from utils.audio import get_audio_bytes
from utils.session import get_state

def ui():
    duration_control = st.slider("Speed Scale", 
                                          value=1.0, 
                                          min_value=0.1, 
                                          max_value=3.0, 
                                          help="Speech speed. Larger value become slow")

    f0_contrl = st.slider("Pitch Scale", 
                                  value=0.333, 
                                  min_value=0.0, 
                                  max_value=1.0)
    
    energy_control = st.slider("Energy Scale", 
                                           value=0.333, 
                                           min_value=0.0, 
                                           max_value=1.0)

    text = st.text_area("Text", value="Enter text for synthesis", height=300, max_chars=2048)

    autoplay_onoff = st.checkbox("Auto play")

    model_not_loaded = st.session_state['text2speech'] is None


    if st.button("Synth!", disabled=model_not_loaded):
        module = st.session_state["text2speech"]

    with torch.no_grad():
        ts = module(text)["wav"]

        wavdata = ts.view(-1).cpu().numpy()

        st.session_state["wavdata"] = wavdata


    if "wavdata" in st.session_state:
        wavdata = get_state(st, 'wavdata')
        samplerate = st.session_state["text2speech"].fs
        st.audio(wavdata, sample_rate=samplerate)

        if autoplay_onoff is True:
            audio_base64 = get_audio_bytes(st, wavdata, samplerate)
            audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
            st.markdown(audio_tag, unsafe_allow_html=True)

        with st.expander("Waveform visualization"):
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(wavdata)

            ax2 = fig.add_subplot(2, 1, 2)
            ax2.specgram(wavdata, Fs=samplerate)
        
            st.pyplot(fig)
