# app.py
import os
import io
import streamlit as st
from transformers import pipeline
# Do NOT import audiocraft at top-level (lazy import later)
from bark.generation import preload_models
from bark import generate_audio, SAMPLE_RATE
from scipy.io.wavfile import write as write_wav
import torch
import torchaudio
import numpy as np
from torch.serialization import add_safe_globals
from time import sleep

# --- Environment: force CPU-only / avoid CuPy paths (helpful on hosts without CUDA)
# This helps avoid thinc trying to load cupy/cuda backends on machines without GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CUPY_DISABLE", "1")

# --- Utility: convert WAV -> MP3 bytes using pydub (import inside function to reduce top-level imports)
def wav_to_mp3_bytes(wav_path: str) -> io.BytesIO:
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(wav_path)
    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3")
    mp3_io.seek(0)
    return mp3_io

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Load Bark models, GPT-2 lyrics generator and MusicGen lazily.
    If audiocraft import fails, this function raises a clear error message.
    """
    # Make torch.load compatible with some saved checkpoints (Bark)
    add_safe_globals([np.core.multiarray.scalar, np.dtype])

    st.info("Loading Bark TTS models...")
    # Patch torch.load to ensure compatibility with weights_only option
    original_torch_load = torch.load
    def safe_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = safe_torch_load

    preload_models()

    st.info("Loading GPT-2 for lyrics generation...")
    lyrics_gen = pipeline("text-generation", model="gpt2")

    # Lazy import MusicGen (audiocraft). This avoids importing spaCy/thinc at module import-time.
    st.info("Importing MusicGen (audiocraft)...")
    try:
        from audiocraft.models import MusicGen
    except Exception as e:
        # Provide a helpful, actionable error
        st.error(
            "Failed to import audiocraft.models.MusicGen. This commonly indicates a binary "
            "incompatibility between NumPy and compiled extensions (thinc/spaCy/C-extensions).\n\n"
            "Actionable steps:\n"
            " - Ensure your requirements.txt matches the pinned versions (numpy==1.24.4, spacy==3.7.4, thinc==8.2.3)\n"
            " - In Streamlit Cloud: Manage app → Settings → Advanced → Clear cache & dependencies, then redeploy\n"
            " - Check the build logs: confirm numpy version installed during build"
        )
        # Re-raise so the build log contains the stack trace
        raise

    st.info("Loading pretrained MusicGen model (facebook/musicgen-small)...")
    music_model = MusicGen.get_pretrained("facebook/musicgen-small")

    return lyrics_gen, music_model

# Load models (cached)
lyrics_generator, music_model = load_models()

# --- Streamlit UI
st.title("🎵 AI Music & Lyrics Generator (Streamlit)")

option = st.radio("Select what to generate:", ("Lyrics Only", "Music Only", "Full Song"))
user_prompt = st.text_area("Enter your prompt (topic, mood, style):", height=140)

if option in ("Music Only", "Full Song"):
    duration = st.slider("Duration (seconds):", min_value=5, max_value=180, value=15)
else:
    duration = None

if st.button("Generate") and user_prompt.strip():
    if option == "Lyrics Only":
        with st.spinner("Generating lyrics..."):
            out = lyrics_generator(user_prompt, max_length=150, num_return_sequences=1)
            lyrics = out[0]["generated_text"]
            sleep(1)
        st.subheader("🎤 Generated Lyrics")
        st.text(lyrics)
        st.download_button("Download Lyrics (TXT)", data=lyrics, file_name="lyrics.txt", mime="text/plain")

    elif option == "Music Only":
        # Generate instrumental
        try:
            music_model.set_generation_params(duration=duration)
            with st.spinner(f"Generating instrumental music ({duration}s)..."):
                music_wav = music_model.generate([user_prompt])  # list of tensors
                # Save first sample
                torchaudio.save("music.wav", music_wav[0].cpu(), 32000)
                sleep(1)
            st.audio("music.wav")
            with open("music.wav", "rb") as f:
                st.download_button("Download Music (WAV)", data=f, file_name="music.wav", mime="audio/wav")
            mp3_bytes = wav_to_mp3_bytes("music.wav")
            st.download_button("Download Music (MP3)", data=mp3_bytes, file_name="music.mp3", mime="audio/mpeg")

        except Exception as e:
            st.error(f"Music generation failed: {e}")
            raise

    elif option == "Full Song":
        # Lyrics
        with st.spinner("Generating lyrics..."):
            out = lyrics_generator(user_prompt, max_length=150, num_return_sequences=1)
            lyrics = out[0]["generated_text"]
            sleep(1)
        st.subheader("🎤 Generated Lyrics")
        st.text(lyrics)

        # Vocals (Bark)
        try:
            with st.spinner("Generating vocals (TTS)..."):
                vocals_audio = generate_audio(lyrics)
                write_wav("vocals.wav", SAMPLE_RATE, vocals_audio)
                sleep(1)
        except Exception as e:
            st.error(f"Failed to generate vocals: {e}")
            raise

        # Instrumental
        try:
            music_model.set_generation_params(duration=duration)
            with st.spinner(f"Generating instrumental music ({duration}s)..."):
                music_wav = music_model.generate([user_prompt])
                torchaudio.save("instrumental.wav", music_wav[0].cpu(), 32000)
                sleep(1)
        except Exception as e:
            st.error(f"Instrumental generation failed: {e}")
            raise

        # Mix (pydub)
        from pydub import AudioSegment
        vocals = AudioSegment.from_wav("vocals.wav") - 2
        instrumental = AudioSegment.from_wav("instrumental.wav") - 3
        final_mix = instrumental.overlay(vocals)
        final_mix.export("final_song.wav", format="wav")

        st.audio("final_song.wav")
        with open("final_song.wav", "rb") as f_audio:
            st.download_button("Download Full Song (WAV)", data=f_audio, file_name="full_song.wav", mime="audio/wav")
        mp3_bytes = wav_to_mp3_bytes("final_song.wav")
        st.download_button("Download Full Song (MP3)", data=mp3_bytes, file_name="full_song.mp3", mime="audio/mpeg")
        st.download_button("Download Lyrics (TXT)", data=lyrics, file_name="lyrics.txt", mime="text/plain")
else:
    if st.button("Generate"):
        st.warning("Please provide a prompt.")
