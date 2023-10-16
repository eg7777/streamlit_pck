import librosa
import soundfile
import streamlit as st
from audio_to_text import audio_to_text, audio_to_text_our_model
from audiorecorder import audiorecorder
from scipy.io import wavfile

st.set_page_config(page_title="My webpage", page_icon=":tada:",
                   layout="wide")


st.title("Singaporean English Speech to Text")

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Record an audio:")
    audio_path = "audio.wav"  # Define the path to save the audio file in the current working directory
    audio_recording = audiorecorder("Click to record", "Click to stop recording")

    if len(audio_recording) > 0:
        st.audio(audio_recording.export().read())
        st.write(f"Frame rate: {audio_recording.frame_rate}, Frame width: {audio_recording.frame_width}, Duration: {audio_recording.duration_seconds} seconds") 

        try:
            # Open and process the audio file
            audio_recording.export(audio_path, format="wav")  # Save the recorded audio
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            text = audio_to_text()
            st.subheader("Baseline Transcription:")
            st.write(text)
        except Exception as e:
            st.error(f"Error (Baseline Transcription): {str(e)}")

        try:
            #audio_data, sample_rate = librosa.load(audio_path, sr=None)
            text = audio_to_text_our_model()
            st.subheader("Transcription:")
            st.write(text)
        except Exception as e:
            st.error(f"Error (Transcription): {str(e)}")


with right_column:
    st.markdown(
        f"""
        <img src="https://mustsharenews.com/wp-content/uploads/2019/09/pck-yellow-boots.jpg" width="275" height="600">
        """,
        unsafe_allow_html=True,
    )
