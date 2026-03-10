# MILTRANS – Context-Aware AI Translation Engine
Project Overview

MILTRANS is an AI-powered multilingual translation system designed to translate Hindi content into multiple regional and international languages. The system supports multiple input formats such as text, images, audio, video, and web URLs.

The application automatically extracts Hindi text from different sources, processes it using NLP techniques, and translates it into selected languages using a deep learning translation model.

The system also stores translations in a database to avoid redundant processing and enable efficient retrieval.

Problem Statement
Many organizations need to translate Hindi content into multiple languages. However, manual translation is:
Time-consuming
Expensive
Difficult when content comes from multiple sources such as images, audio, or web pages
This project automates the process of extracting and translating Hindi text using AI.

Solution
MILTRANS uses a combination of computer vision, speech recognition, and natural language processing techniques.
The system performs the following steps:
Extract Hindi text from different input formats
Normalize and preprocess the extracted text
Translate the text into multiple languages
Store translations in a database
Allow users to download the translated output
The system is implemented as an interactive web application using Streamlit.

Features
Multi-language translation support
Multiple input formats
Image OCR support
Audio speech recognition
Video speech extraction
Web page text extraction
Translation database storage
Downloadable translated output
Interactive user interface
Supported Input Types

The system can process the following input formats:
Text Input
Text File (.txt)
Image with Hindi text
Web URL
Audio file (MP3 / WAV)
Video file (MP4 / MKV / AVI)

System Workflow
User Input
   ↓
Text / Image / Audio / Video / URL
   ↓
Text Extraction
   ↓
Hindi Text Normalization
   ↓
Translation Model
   ↓
Multi-Language Output
   ↓
Save to Database
   ↓
Download Translated File
Technologies Used

Python
Natural Language Processing
Speech Recognition
Optical Character Recognition

Libraries and Tools:

Streamlit

Hugging Face Transformers

EasyOCR

BeautifulSoup

MongoDB

SpeechRecognition

Pydub

Translation Model

The system uses the NLLB‑200 Distilled 600M multilingual translation model to translate Hindi text into multiple languages.

Supported languages include:

English
Kannada
Tamil
Telugu
Marathi
Malayalam
Punjabi
Gujarati
Urdu
Assamese
Nepali
and others.

Database Integration

Translations are stored using MongoDB Atlas.

Benefits:

Prevents duplicate translations

Improves system efficiency

Allows retrieval of previously translated text

Project Structure
MILTRANS_project

README.md
app.py
requirements.txt
Installation

Clone the repository:

git clone https://github.com/your-username/MILTRANS_project.git

Navigate to the project directory:

cd MILTRANS_project

Install required libraries:

pip install -r requirements.txt
Run the Application

Start the Streamlit application:

streamlit run app.py

After running the command, the application will open in your web browser.

Example Workflow

Upload an image containing Hindi text

The system extracts the text using OCR

The extracted text is normalized

The text is translated into selected languages

The translation is displayed and stored in the database

Future Improvements

Real-time translation for live audio/video streams

Mobile application deployment

Improved speech recognition accuracy

Support for additional languages

Author

Chaitra Patil
Data Science Student – 360DigiTMG
