import io
import base64
import soundfile
import streamlit as st

def get_audio_bytes(audio_bytes, sample_rate):
    """
    audio_bytes : np.ndarray dtype fp32
    """

    buf = io.BytesIO()
    soundfile.write(buf, audio_bytes, samplerate=sample_rate, format='WAV')

    wavdata = buf.getvalue()
    
    # https://github.com/streamlit/streamlit/issues/2446#issuecomment-1465017176
    audio_base64 = base64.b64encode(wavdata).decode('utf-8')

    audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
    st.markdown(audio_tag, unsafe_allow_html=True)

def save_audio(audio_bytes, sample_rate, file_path):
    """
    Save the file to a filepath
    """
    try:
        soundfile.write(file_path, audio_bytes, samplerate=sample_rate, format="WAV")
    except Exception as e:
        st.error("File not saved")
        print(e)
