import streamlit as st
from transformers import pipeline
from audiocraft.models import MusicGen
from bark.generation import preload_models
from bark import generate_audio, SAMPLE_RATE
from scipy.io.wavfile import write as write_wav
import torch
import torchaudio
import numpy as np
from torch.serialization import add_safe_globals
from time import sleep
from pydub import AudioSegment
import io


@st.cache_resource
def load_models():
    import numpy as np
    from torch.serialization import add_safe_globals
    add_safe_globals([np.core.multiarray.scalar, np.dtype])

    st.info("Loading Bark speech synthesis models...")

    # Patch torch.load safely
    original_torch_load = torch.load
    def safe_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = safe_torch_load

    preload_models()

    st.info("Loading GPT-2 for lyrics...")
    lyrics_gen = pipeline("text-generation", model="gpt2")

    st.info("Loading MusicGen model...")
    music_model = MusicGen.get_pretrained("facebook/musicgen-small")

    return lyrics_gen, music_model


lyrics_generator, music_model = load_models()

st.title("🎵 AI Music & Lyrics Generator")

option = st.radio("Select what you want to generate:", ("Lyrics Only", "Music Only", "Full Song"))

user_prompt = st.text_area("Enter your prompt (e.g., topic, mood, style):", height=120)

if option in ["Music Only", "Full Song"]:
    duration = st.slider("Select duration (seconds):", min_value=5, max_value=180, value=15)
else:
    duration = None  # No music duration for lyrics-only

def wav_to_mp3_bytes(wav_path):
    audio = AudioSegment.from_wav(wav_path)
    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3")
    mp3_io.seek(0)
    return mp3_io

if st.button("Generate") and user_prompt.strip() != "":
    if option == "Lyrics Only":
        with st.spinner("Generating lyrics..."):
            lyrics = lyrics_generator(user_prompt, max_length=150, num_return_sequences=1)[0]["generated_text"]
            sleep(1)
        st.subheader("🎤 Generated Lyrics")
        st.text(lyrics)

        st.download_button(
            label="Download Lyrics as TXT",
            data=lyrics,
            file_name="lyrics.txt",
            mime="text/plain"
        )

    elif option == "Music Only":
        music_model.set_generation_params(duration=duration)
        with st.spinner(f"Generating instrumental music ({duration}s)..."):
            music_wav = music_model.generate([user_prompt])
            torchaudio.save("music.wav", music_wav[0].cpu(), 32000)
            sleep(1)
        st.audio("music.wav")

        # WAV download
        with open("music.wav", "rb") as f:
            st.download_button("Download Music (WAV)", data=f, file_name="music.wav", mime="audio/wav")

        # MP3 download
        mp3_bytes = wav_to_mp3_bytes("music.wav")
        st.download_button("Download Music (MP3)", data=mp3_bytes, file_name="music.mp3", mime="audio/mpeg")

    elif option == "Full Song":
        music_model.set_generation_params(duration=duration)
        with st.spinner("Generating lyrics..."):
            lyrics = lyrics_generator(user_prompt, max_length=150, num_return_sequences=1)[0]["generated_text"]
            sleep(1)
        st.subheader("🎤 Generated Lyrics")
        st.text(lyrics)

        with st.spinner("Generating vocals..."):
            vocals_audio = generate_audio(lyrics)
            write_wav("vocals.wav", SAMPLE_RATE, vocals_audio)
            sleep(1)

        with st.spinner(f"Generating instrumental music ({duration}s)..."):
            music_wav = music_model.generate([user_prompt])
            torchaudio.save("instrumental.wav", music_wav[0].cpu(), 32000)
            sleep(1)

        vocals = AudioSegment.from_wav("vocals.wav")
        instrumental = AudioSegment.from_wav("instrumental.wav")

        vocals = vocals - 2
        instrumental = instrumental - 3
        final_mix = instrumental.overlay(vocals)
        final_mix.export("final_song.wav", format="wav")

        st.audio("final_song.wav")

        # WAV download
        with open("final_song.wav", "rb") as f_audio:
            st.download_button("Download Full Song (WAV)", data=f_audio, file_name="full_song.wav", mime="audio/wav")

        # MP3 download
        mp3_bytes = wav_to_mp3_bytes("final_song.wav")
        st.download_button("Download Full Song (MP3)", data=mp3_bytes, file_name="full_song.mp3", mime="audio/mpeg")

        st.download_button(
            label="Download Lyrics as TXT",
            data=lyrics,
            file_name="lyrics.txt",
            mime="text/plain"
        )

    else:
        st.write("Please select an option and provide a prompt.")
