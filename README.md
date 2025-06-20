# 🎵 AI Music & Lyrics Generator (Streamlit + Colab)

This project is an end-to-end **AI-powered music generation system** that allows **anyone**—musicians, content creators, educators, or hobbyists—to generate:

- Custom song lyrics
- AI-generated instrumental music
- Full songs with AI vocals, lyrics, and background music

Built to run fully in **Google Colab** using **Streamlit + Ngrok**, this project brings cutting-edge **Generative AI** directly to your browser with no installation required.

---

## Why This Project?

- **No musical background needed** — just type what you want!
- Generate complete songs in **MP3/WAV** format
- Uses powerful models: GPT-2, MusicGen, and Bark
- Runs completely in the cloud (Colab GPU), no downloads
- Great for learning how generative AI works in music, lyrics, and TTS

---

## Who Can Use This?

| User Type           | How It Helps                                                           |
|---------------------|------------------------------------------------------------------------|
| Musicians           | Quickly prototype song ideas and lyrics                                |
| Developers          | Learn to combine AI models into real apps                              |
| Students            | Understand how text, sound, and models can be fused into music AI     |
| Content Creators    | Generate background music, podcast openers, YouTube audio intros      |
| Researchers         | Explore open-source audio AI models in a usable environment           |

---

## Features at a Glance

- **User Options**: Lyrics only, Music only, Full Song
- **Lyrics Generation**: GPT-2 writes thematic song lyrics
- **Music Generation**: MusicGen creates realistic AI music in seconds
- **Voice Synthesis**: Bark converts lyrics to AI vocals
- **Mixing Engine**: Merges vocals + background music
- **Download Options**: MP3 and WAV for audio, TXT for lyrics
- **Flexible Duration**: Music/songs from 5s to 180s
- **No Setup Needed**: Fully works in a Colab notebook

---

## How It Works

1. User selects what they want to generate: Lyrics / Music / Full Song
2. Provides a **natural language prompt** like:
   > "Create a pop love song about the rain"
3. Based on the choice:
   - Lyrics are generated by **GPT-2**
   - Music is generated by **MusicGen**
   - AI voice is created using **Bark**
4. If Full Song selected: lyrics & music are **mixed** into one file
5. Users can  preview +  download the result in **WAV/MP3/TXT**

---

## Prompt Examples

| Type        | Example Prompt                                                   |
|-------------|------------------------------------------------------------------|
| Lyrics      | "Write a sad love song about lost time"                         |
| Music       | "Lo-fi chill instrumental for studying in the evening"           |
| Full Song   | "Upbeat dance song with female vocals about summer vacations"    |

---

## App UI (Streamlit)

- Prompt text input
- Duration slider (5–180 seconds)
- Option selection (Lyrics, Music, Full Song)
- Audio player for results
- Download buttons (MP3/WAV/TXT)

---

## Setup & Run (in Colab)

1. *Install all required packages*:

```bash
!pip install streamlit transformers torchaudio audiocraft git+https://github.com/suno-ai/bark.git pydub pyngrok
!apt-get install -y ffmpeg
```

2. **Create the app.py file** with the full Streamlit logic (already included).

3. **Run the app using ngrok to expose Streamlit UI**:

```python
from pyngrok import ngrok
!kill $(pgrep streamlit) || echo "No streamlit running"
get_ipython().system_raw("streamlit run app.py &")
public_url = ngrok.connect(port="8501")
print(f"🔗 Open your app here: {public_url}")
```

---

## Output Formats

| Output Type     | File Format  |
|------------------|--------------|
| Lyrics           | .txt       |
| Music / Song     | .wav, .mp3 |

---

## Example Use Cases

- **Songwriters** generating starter lyrics and melody
- **YouTubers** creating background audio without copyright risk
- **App developers** testing generative audio systems
- **Students** learning multi-modal AI with real examples

---

## Future Enhancements (You can contribute!)

- Voice type selector (e.g., male/female/robotic)
- Volume & tempo controls
- Style presets (trap, jazz, EDM, rock, cinematic)
- Multi-section song builder (intro, verse, chorus, outro)

---

## Credits

- [OpenAI GPT-2 (Lyrics)](https://huggingface.co/gpt2)
- [Meta MusicGen (Music)](https://github.com/facebookresearch/audiocraft)
- [Bark TTS by Suno (Voice)](https://github.com/suno-ai/bark)
- [Streamlit](https://streamlit.io/)
- [PyDub](https://github.com/jiaaro/pydub)
- [Ngrok](https://ngrok.com/)
-
