import openai
import os

API_KEY = "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"


def api_setup():
  # Hackathon key:
  # API_KEY = "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"
  # Josh key
  # API_KEY = "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"
  # Clay key
  openai.api_key = API_KEY
