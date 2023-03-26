import requests
import tempfile
import openai
from openai import InvalidRequestError
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

import logging

log = logging.getLogger(__name__)


def get_silent_time(audio_file_path):
  audio_file = AudioSegment.from_wav(audio_file_path)

  silence_chunks = split_on_silence(
    audio_file,
    min_silence_len=1000,  # minimum length of silence in milliseconds
    silence_thresh=-52)  # threshold for silence (in dBFS)

  total_time = 0
  for i, chunk in enumerate(silence_chunks):
    total_time += chunk.duration_seconds
  return total_time


def get_num_fillers(audio_file_path):
  import speech_recognition as sr
  import re

  filler_words = ["um", "uh", "ah", "like"]
  filler_words_regex = r"\b(" + "|".join(filler_words) + r")\b"

  audio_file = sr.AudioFile(audio_file_path)
  recognizer = sr.Recognizer()

  with audio_file as source:
    audio = recognizer.record(source)

  text = recognizer.recognize_google(audio)
  matches = re.findall(filler_words_regex, text, re.IGNORECASE)
  num_filler_words = len(matches)
  return num_filler_words


RACHEL_VOICE = '21m00Tcm4TlvDq8ikWAM'


def synth_text_as_file(text):
  # Download the MPEG file from the URL
  url = f"https://api.elevenlabs.io/v1/text-to-speech/{RACHEL_VOICE}"
  log.debug(f'calling TTS API with text: {text}')
  response = requests.post(url,
                           headers={
                             'xi-api-key': '7c9d07690c72866906179decb3bc9dd3',
                             'accept': 'audio/mpeg',
                             'Content-Type': 'application/json'
                           },
                           json={
                             "text": text,
                             "voice_settings": {
                               "stability": 0,
                               "similarity_boost": 0
                             }
                           },
                           timeout=120)
  if response.status_code == 200:
    # create a temporary file in the system's default temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
      # write some data to the file
      temp_file.write(response.content)
      return temp_file.name
  else:
    raise Exception(
      f"Error: API request returned {response.status_code} status code")


def save_audio(bytes):
  with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
    temp_file.write(bytes)
    temp_file.flush()
    os.fsync(temp_file.fileno())

    #os.shell('python extract_pauses_1.py n y')
    return temp_file.name


def transcribe_speech(audio_file):
  log.debug('transcribing audio...')
  with open(audio_file, 'rb') as f:
    try:
      transcript = openai.Audio.transcribe("whisper-1", f)
    except InvalidRequestError:
      return None
  return transcript.text
