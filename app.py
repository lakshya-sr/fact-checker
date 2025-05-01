from flask import Flask, request, render_template
import subprocess
import os
import uuid
import yt_dlp
import torch
from transformers import pipeline
import requests
from langdetect import detect

# Initialize Flask app
app = Flask(__name__)

# Replace this with your actual Google API Key
GOOGLE_FACT_CHECK_API_KEY = 'AIzaSyAZyWTbPrJfuQfjZK2bGBSgGSYacZJdumc'

# Load Huggingface pipelines
device = 0 if torch.cuda.is_available() else -1
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
translation_pipeline = pipeline("translation", model="facebook/m2m100_418M", device=device)


# Helper: download audio from video
def download_audio(video_url):
    output_path = f"/tmp/{uuid.uuid4()}.mp3"
    ydl_opts = {
        # 'username': 'soogandisnut',
        # 'password': 'SOO6$8FkEM#9HQ',
        'format': 'bestaudio/best',
        'outtmpl': '/tmp/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        downloaded_path = ydl.prepare_filename(info).rsplit('.', 1)[0] + '.mp3'
    return downloaded_path

# Helper: transcribe audio to text
def transcribe_audio(audio_path):
    result = asr_pipeline(audio_path, return_timestamps=True)
    return result['text']

# Helper: translate text to English
def translate_text(text):
    detected_lang = detect(text)
    print(f"Detected language: {detected_lang}")

    if detected_lang == 'en':
        return text  # No need to translate
    
    # Map langdetect language code to M2M100 language code (they match for most)
    translated = translation_pipeline(
        text,
        src_lang=detected_lang,
        tgt_lang="en"
    )
    return translated[0]['translation_text']

# Helper: summarize text
def summarize_text(text):
    summarized = summarization_pipeline(text, max_length=150, min_length=50, do_sample=False)
    return summarized[0]['summary_text']

# Helper: query fact check API
def fact_check(statements):
    results = {statement:[] for statement in statements}
    for statement in statements:
        url = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'
        params = {
            'query': statement,
            'key': GOOGLE_FACT_CHECK_API_KEY
        }
        response = requests.get(url, params=params)
        claims = []
        if response.status_code == 200:
            data = response.json()
            if 'claims' in data:
                for claim in data['claims']:
                    claim_info = {
                        'claim_text': claim.get('text', 'No text'),
                        'rating': claim.get('claimReview', [{}])[0].get('textualRating', 'Unknown'),
                        'publisher': claim.get('claimReview', [{}])[0].get('publisher', {}).get('name', 'Unknown'),
                        'url': claim.get('claimReview', [{}])[0].get('url', '#')
                    }
                    claims.append(claim_info)
        results[statement].append(claims)
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_url = request.form['video_url']
        # video_url = 'https://www.instagram.com/p/DI8PkcDs45b/'
        audio_path = download_audio(video_url)
        print(audio_path)
        transcript = transcribe_audio(audio_path)
        print(transcript)
        translated_text = translate_text(transcript)
        print(translated_text)
        summary = summarize_text(translated_text)
        print(summary)
        statements = get_statements(summary)
        claims = fact_check(statements)
        print(claims)
        os.remove(audio_path)  # cleanup
        return render_template("index.html", result={'summary': summary, 'claims': claims if claims else [{'claim_text': 'No claims found'}]})
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
    # index()