import streamlit as st
import easyocr
from PIL import Image
import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import numpy as np
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment
from pydub.utils import which
from audio_recorder_streamlit import audio_recorder
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import tempfile
import os

# ---------------------------
# Configure pydub to use ffmpeg
# ---------------------------
AudioSegment.converter = which("ffmpeg")

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="MILTRANS: Hindi → All Languages", page_icon="🌐")

# ---------------------------
# MongoDB Atlas Connection
# ---------------------------
username = "patilchaitra612"
password = "12345"  # ⚠️ For production, store securely (env vars)
uri = f"mongodb+srv://{username}:{password}@cluster1.oxcvzfp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"

client = MongoClient(uri, server_api=ServerApi('1'))
try:
    client.admin.command('ping')
    st.sidebar.success("✅ Connected to MongoDB Atlas")
except Exception as e:
    st.sidebar.error(f"❌ MongoDB connection failed: {e}")

db = client["translations_db"]
collection = db["translations"]

def save_translation(source_text, translated_text, source_lang="hi", target_lang="en"):
    """Save translation to MongoDB if it doesn't already exist."""
    existing_doc = collection.find_one({
        "source_text": source_text,
        "source_lang": source_lang,
        "target_lang": target_lang
    })
    if existing_doc:
        st.sidebar.info(f"ℹ️ Translation for {target_lang} already exists in DB.")
        return existing_doc
    new_doc = {
        "source_text": source_text,
        "translated_text": translated_text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    collection.insert_one(new_doc)
    st.sidebar.success(f"✅ Saved new translation ({target_lang}) to DB.")
    return new_doc

# ---------------------------
# Supported Languages (NLLB codes)
# ---------------------------
LANGUAGES = {
    "English": "eng_Latn",
    "Kannada": "kan_Knda",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Marathi": "mar_Deva",
    "Malayalam": "mal_Mlym",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Urdu": "urd_Arab",
    "Gujarati": "guj_Gujr",
    "Assamese": "asm_Beng",
    "Bhojpuri": "bho_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Magahi": "mag_Deva",
    "Maithili": "mai_Deva",
    "Nepali": "npi_Deva",
    "Manipuri (Meitei)": "mni_Beng"
}

# ---------------------------
# Sidebar - select multiple target languages (English always included)
# ---------------------------
target_languages = st.sidebar.multiselect(
    "Select target languages:",
    options=list(LANGUAGES.keys()),
    default=["English"]
)
if "English" not in target_languages:
    target_languages.insert(0, "English")

# ---------------------------
# Initialize EasyOCR Reader
# ---------------------------
reader = easyocr.Reader(['hi'])

# ---------------------------
# Hindi Normalizer
# ---------------------------
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("hi")

def normalize_hindi(text):
    # Clean + normalize
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return normalizer.normalize(text)

# ---------------------------
# Load translator (cached per src/tgt combo)
# ---------------------------
@st.cache_resource
def load_nllb_translator(src_code, tgt_code):
    return pipeline(
        "translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang=src_code,
        tgt_lang=tgt_code
    )

def translate_text(text, tgt_lang_code):
    translator = load_nllb_translator("hin_Deva", tgt_lang_code)
    result = translator(text)
    return result[0]['translation_text']

# ---------------------------
# Speech-to-text helper (shared by Audio & Video)
# ---------------------------
def _speech_to_text_from_audiosegment(audio_segment: AudioSegment) -> str:
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8

    # Convert to mono 16k PCM WAV for best ASR results
    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
    wav_io = BytesIO()
    audio_segment.export(wav_io, format="wav", parameters=["-acodec", "pcm_s16le"])
    wav_io.seek(0)

    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)
        if len(audio_data.frame_data) == 0:
            return "Error: No speech detected in the audio."
        try:
            return recognizer.recognize_google(audio_data, language="hi-IN")
        except sr.UnknownValueError:
            return "Error: Could not understand audio."
        except sr.RequestError as e:
            return f"Error: Google Speech Recognition request failed; {e}"

# ---------------------------
# OCR Function
# ---------------------------
def extract_text_from_image(image_file):
    img = Image.open(image_file)
    result = reader.readtext(np.array(img), detail=0)
    text = " ".join(result).strip()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^\w\s\.,;!?-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ---------------------------
# Web scraping
# ---------------------------
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=20)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'[^\w\s\.,;!?-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {e}"

# ---------------------------
# Audio to text
# ---------------------------
def extract_text_from_audio(audio_file):
    try:
        audio_segment = AudioSegment.from_file(audio_file)
        return _speech_to_text_from_audiosegment(audio_segment)
    except Exception as e:
        return f"Error extracting text from audio: {e}"

# ---------------------------
# NEW: Video to text (video -> audio -> text)
# ---------------------------
def extract_text_from_video(video_file):
    """
    Try using pydub+ffmpeg directly. If that fails for some container/codecs,
    fallback to moviepy to extract WAV, then run ASR.
    """
    # 1) Try with pydub (works for many mp4/mkv if ffmpeg can decode)
    try:
        audio_seg = AudioSegment.from_file(video_file)  # ffmpeg handles container
        return _speech_to_text_from_audiosegment(audio_seg)
    except Exception as e_pydub:
        # 2) Fallback: moviepy
        try:
            from moviepy.editor import VideoFileClip
            # VideoFileClip needs a filename; write bytes to temp file if needed
            if hasattr(video_file, "read"):
                # Streamlit UploadedFile or BytesIO
                raw = video_file.read()
                tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tmp_in.write(raw)
                tmp_in.flush()
                tmp_in.close()
                video_path = tmp_in.name
            else:
                # already a path-like
                video_path = video_file

            clip = VideoFileClip(video_path)
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            clip.audio.write_audiofile(tmp_wav, fps=16000, nbytes=2, codec="pcm_s16le", verbose=False, logger=None)
            clip.close()

            # Load WAV and run ASR
            audio_seg = AudioSegment.from_wav(tmp_wav)
            text = _speech_to_text_from_audiosegment(audio_seg)

            # cleanup
            try:
                os.remove(tmp_wav)
                if 'tmp_in' in locals():
                    os.remove(tmp_in.name)
            except Exception:
                pass

            return text
        except Exception as e_moviepy:
            return f"Error extracting text from video. pydub error: {e_pydub}; moviepy error: {e_moviepy}"

# ---------------------------
# UI
# ---------------------------
st.title("🎖️ MILTRANS: Context-Aware AI Translation Engine for Military SOPs")

input_type = st.radio(
    "Choose input type:",
    ["Text", "File (.txt)", "Image", "Web URL", "Audio", "Video"],
    index=0,
    horizontal=True
)

if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "editable_text" not in st.session_state:
    st.session_state.editable_text = ""

hindi_text = ""

# ---------------------------
# Input Handling
# ---------------------------
if input_type == "Text":
    st.session_state.text_input = st.text_area(
        "Enter Hindi text:",
        value=st.session_state.text_input,
        height=150
    )
    hindi_text = st.session_state.text_input

elif input_type == "File (.txt)":
    uploaded_file = st.file_uploader("Upload Hindi text file", type=["txt"])
    if uploaded_file:
        extracted_text = uploaded_file.read().decode("utf-8")
        st.session_state.editable_text = st.text_area(
            "Edit extracted text:",
            value=extracted_text,
            height=150
        )
        hindi_text = st.session_state.editable_text

elif input_type == "Image":
    uploaded_image = st.file_uploader("Upload image with Hindi text", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        extracted_text = extract_text_from_image(uploaded_image)
        st.session_state.editable_text = st.text_area(
            "Edit extracted text:",
            value=extracted_text,
            height=150
        )
        hindi_text = st.session_state.editable_text

elif input_type == "Web URL":
    url_input = st.text_input("Enter Web URL:")
    if url_input:
        extracted_text = extract_text_from_url(url_input)
        st.session_state.editable_text = st.text_area(
            "Edit extracted text:",
            value=extracted_text,
            height=150
        )
        hindi_text = st.session_state.editable_text

elif input_type == "Audio":
    st.info("Upload an audio file or record from mic.")
    audio_option = st.radio("Audio input type:", ["Upload File", "Record from Mic"], index=0, horizontal=True)

    if audio_option == "Upload File":
        uploaded_audio = st.file_uploader("Upload audio (mp3/wav)", type=["mp3", "wav"])
        if uploaded_audio:
            extracted_text = extract_text_from_audio(uploaded_audio)
            st.session_state.editable_text = st.text_area(
                "Edit extracted text:",
                value=extracted_text,
                height=150
            )
            hindi_text = st.session_state.editable_text

    elif audio_option == "Record from Mic":
        st.caption("Tip: Speak clearly, keep the mic close, and pause briefly at the end.")
        audio_bytes = audio_recorder()
        if audio_bytes:
            audio_file = BytesIO(audio_bytes)
            extracted_text = extract_text_from_audio(audio_file)
            st.session_state.editable_text = st.text_area(
                "Edit extracted text:",
                value=extracted_text,
                height=150
            )
            hindi_text = st.session_state.editable_text

elif input_type == "Video":
    st.info("Upload a Hindi video; we’ll extract speech and translate it.")
    uploaded_video = st.file_uploader("Upload video (mp4/mkv/avi/mov/webm)", type=["mp4","mkv","avi","mov","webm"])
    if uploaded_video:
        with st.spinner("Extracting audio and transcribing Hindi speech..."):
            extracted_text = extract_text_from_video(uploaded_video)
        st.session_state.editable_text = st.text_area(
            "Edit extracted text:",
            value=extracted_text,
            height=150
        )
        hindi_text = st.session_state.editable_text

# ---------------------------
# Translate Button
# ---------------------------
if st.button("Translate"):
    if hindi_text.strip():
        normalized_text = normalize_hindi(hindi_text)

        translations = {}
        for lang in target_languages:
            with st.spinner(f"Translating to {lang}..."):
                translated_text = translate_text(normalized_text, LANGUAGES[lang])
                translations[lang] = translated_text
                save_translation(normalized_text, translated_text, "hi", lang)

        st.subheader("✅ Translated Texts:")
        for lang, text in translations.items():
            st.markdown(f"**{lang}:** {text}")

        # Download all results
        download_data = f"Original Hindi Text:\n{normalized_text}\n\n"
        for lang, text in translations.items():
            download_data += f"{lang}:\n{text}\n\n"

        st.download_button(
            label="⬇️ Download All Translations",
            data=download_data.encode("utf-8"),
            file_name="translations.txt",
            mime="text/plain"
        )

    else:
        st.warning("Please provide text input.")
