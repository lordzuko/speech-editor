import torch
import streamlit as st
import matplotlib.pyplot as plt
from utils.audio import get_audio_bytes
from utils.session import get_state, init_session_state
from fs2.controlled_synthesis import  preprocess_single, synthesize
from config import lexicon, g2p, args, preprocess_config, configs



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
    
    model_not_loaded = get_state(st, "model") is None


    if st.button("Synth!", disabled=model_not_loaded):
        model = get_state(st, "model")

    with torch.no_grad():
        control_values, batchs = preprocess_single(text, lexicon, g2p, args, preprocess_config)
        wavdata = synthesize(get_state(st, "model"), configs, get_state(st, "vocoder"), batchs, control_values)

        # ts = get_state(st, "model")(text)
        # wavdata = ts.view(-1).cpu().numpy()
        init_session_state(st, "wavdata", wavdata)

    if "wavdata" in st.session_state:
        wavdata = get_state(st, 'wavdata')
        samplerate = get_state(st, 'sampling_rate')
        st.audio(wavdata, sample_rate=samplerate)

        if autoplay_onoff is True:
            audio_base64 = get_audio_bytes(wavdata, samplerate)
            audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
            st.markdown(audio_tag, unsafe_allow_html=True)

        with st.expander("Waveform visualization"):
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(wavdata)

            ax2 = fig.add_subplot(2, 1, 2)
            ax2.specgram(wavdata, Fs=samplerate)
        
            st.pyplot(fig)
