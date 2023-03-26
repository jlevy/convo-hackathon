import openai
import os


def api_setup():
  # Hackathon key:
  # API_KEY = "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"
  # Josh key
  # API_KEY = "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"
  # Clay key
  API_KEY = "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"
  openai.api_key = API_KEY
