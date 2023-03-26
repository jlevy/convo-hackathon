import os
import json
from collections import namedtuple
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  AIMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
import openai
from mutagen.mp3 import MP3
import librosa
from speech_handling import get_num_fillers, get_silent_time

os.environ[
  "OPENAI_API_KEY"] = "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"

Scenario = namedtuple(
  "Scenario",
  ["name", "description", "convo_prompt_template", "eval_prompt_template"])

INTERVIEW_PROMPT_TEMPLATE = """You are a hiring manager looking for the best candidate for a job. 
Your job is to learn more about the candidate and ask relevant questions 
to learn more about them to determine if they are a good fit for the job.
You should only ask one question at a time. You should consider the candidate's job history:

Candidate job history: {job_history}

The job is: {job_type}

Conversation history: 
{history}
----
"""

PARTY_PROMPT_TEMPLATE = """You are a human at a party and speaking with another person you don't know very well. 
Your should make pleasant conversation to know the other person better.
You should only ask the other person one question at a time.

Conversation history: 
{history}
----
"""

CALFRESH_PROMPT_TEMPLATE = """You are a representative for CalFresh and are interviewing an applicant for benefit eligibility.

Your job is to learn more about the candidate and ask relevant questions 
to learn more about them to determine if they are elegible for benefits.
You should only ask one question at a time.

Conversation history: 
{history}
----
"""

CONGRESS_PROMPT_TEMPLATE = """You are a powerful US Senator from a large state. Your job is to grandstand, ask questions,
and advance a murky political agenda.

Ask the candidate hostile questions about the regulation of AI technology, while demonstrating no real
knowledge of the technology. The questions should be short buy vauge. Address the canidate as "Whoever you are".

Conversation history: 
{history}
----
"""

FIRST_DATE_TEMPLATE = """You are a human on a first date.

Ask the the other person questions about their background and what motivates then. Only ask one question at a time,
and be friendly, engaged, and mildly flirty.

Conversation history: 
{history}
----
"""

AI_HACKATHON_PROMPT_TEMPLATE = """You are a judge at an AI Hackathon focused on social good.

Ask the candidate detailed questions about what they have built, specifically around how it helps society, how creative it is,
the underlying technology and innovation. You should only ask one question at a time.

Conversation history: 
{history}
----
"""

EVALUATE_PROMPT_TEMPLATE = """
Evaluate the following transcript from an in-person conversation for the following:

* Clarity on a scale of 1-10, where 1 is the candidate was overly verbose and did not answer the question at a all and 10 is the candidate answered the questions directly, in few words, and answered the question with relevant examples.
* Confidence on a scale of 1-10, where 1 is the candidate was uncertain, hesitant, and overly self-deprecating and 10 where the candidate was clear and described their accomplishments.
* Vocabulary on a scale of 1-10, there 1 is the candidate always chose the wrong word and used incomprehensible English, or 10 where the candidate used advanced language and terminology.
* Poise on a scale of 1-10, where 1 is the candidate seemed very awkward and 10 is the candidate seemed poised and socially aware.

You must return the evaluation scores and summaryt in a JSON format with the keys "clarity", "confidence", "vocabulary", "poise", and "summary". You MUST not return anything other than a single, valid JSON object.

—

Transcript:
{transcript}
"""

SCENARIOS = [
  Scenario(
    "Bigco Job Interview",
    "You have an introductory interview with an interviewer at a large company.",
    INTERVIEW_PROMPT_TEMPLATE, EVALUATE_PROMPT_TEMPLATE),
  Scenario(
    "Party Chitchat",
    "You're at an event and start an unexpected conversation with someone you've never met.",
    PARTY_PROMPT_TEMPLATE, EVALUATE_PROMPT_TEMPLATE),
  Scenario(
    "CalFresh Benefit Interview",
    "You want to prepare for interviewing with CalFresh for benefit eligibility.",
    CALFRESH_PROMPT_TEMPLATE, EVALUATE_PROMPT_TEMPLATE),
  Scenario("First Date", "You want to prepare for a first date.",
           FIRST_DATE_TEMPLATE, EVALUATE_PROMPT_TEMPLATE),
  Scenario(
    "Testify Before Congress",
    "You want to prepare a hostile interview in Congress about your company's monopoly.",
    CONGRESS_PROMPT_TEMPLATE, EVALUATE_PROMPT_TEMPLATE),
  Scenario("Pitch at an AI Hackathon",
           "Prepare a demo pitch at an AI hackaton.",
           AI_HACKATHON_PROMPT_TEMPLATE, EVALUATE_PROMPT_TEMPLATE)
]

import logging

log = logging.getLogger(__name__)


def build_scenario_prompt(template):
  system_message_prompt = SystemMessagePromptTemplate.from_template(template)
  human_template = "{text}"
  human_message_prompt = HumanMessagePromptTemplate.from_template(
    human_template)

  return ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])


PARTY_PROMPT = build_scenario_prompt(PARTY_PROMPT_TEMPLATE)
INTERVIEW_PROMPT = build_scenario_prompt(INTERVIEW_PROMPT_TEMPLATE)
CALFRESH_PROMPT = build_scenario_prompt(CALFRESH_PROMPT_TEMPLATE)
EVALUATE_PROMPT = build_scenario_prompt(EVALUATE_PROMPT_TEMPLATE)
CONGRESS_PROMPT = build_scenario_prompt(CONGRESS_PROMPT_TEMPLATE)
HACKATHON_PROMPT = build_scenario_prompt(AI_HACKATHON_PROMPT_TEMPLATE)
FIRST_DATE_PROMPT = build_scenario_prompt(FIRST_DATE_TEMPLATE)

MAX_CONVO_CNT = 3

chat = ChatOpenAI(temperature=0, model='gpt-4')


class ConversationState:
  """The state of a single conversation."""

  def __init__(self, scenario_name):
    self.history = []
    self.audio_history = []
    self.scenario_name = scenario_name
    self.last_user_input = None
    self.last_ai_response = None
    self.active = False
    self.score = None

  def format_history(self, history):
    output = ""
    for chat in history:
      human, system = chat
      output = output + f"Human: {human}\nSystem: {system}\n"
    return output

  def history_for_prompt(self):
    return self.format_history(self.history)

  def greet_user(self, context):
    self.handle_message("", None, context)
    self.active = True

  def handle_message(self, user_input, audio_file, context):
    log.info(f"Constructing prompt with context: {context}")
    if user_input == self.last_user_input:
      return False
    self.last_user_input = user_input
    messages = []
    if self.scenario_name == "Bigco Job Interview":
      if context == "":
        context = "unknown job history"
      messages = INTERVIEW_PROMPT.format_prompt(job_type=context,
                                                job_history=context,
                                                history=self.format_history(
                                                  self.history),
                                                text=user_input).to_messages()
    elif self.scenario_name == "Party Chitchat":
      messages = PARTY_PROMPT.format_prompt(history=self.format_history(
        self.history),
                                            text=user_input).to_messages()
    elif self.scenario_name == "CalFresh Benfit Interview":
      messages = CALFRESH_PROMPT.format_prompt(history=self.format_history(
        self.history),
                                               text=user_input).to_messages()
    elif self.scenario_name == "Testify Before Congress":
      messages = CONGRESS_PROMPT.format_prompt(history=self.format_history(
        self.history),
                                               text=user_input).to_messages()
    elif self.scenario_name == "Pitch at an AI Hackathon":
      messages = HACKATHON_PROMPT.format_prompt(history=self.format_history(
        self.history),
                                                text=user_input).to_messages()
    elif self.scenario_name == "First Date":
      messages = FIRST_DATE_PROMPT.format_prompt(
        history=self.format_history(self.history),
        text=user_input).to_messages()

    log.debug(f'Calling OpenAI with input: ${user_input}')
    output = chat(messages)
    ai_response = output.content
    self.history.append((user_input, ai_response))
    if audio_file:
      self.audio_history.append(audio_file)
    self.last_ai_response = ai_response
    return True

  def calc_avg_wpm(self, audio_lens):
    transcription_lens = [len(item[0].split()) for item in self.history]
    if len(audio_lens) == len(transcription_lens):
      wpms = [((transcription_lens[i] / audio_lens[i]) * 60)
              for i in range(audio_lens)]
      return sum(wpms) / len(wpms)
    else:
      log.warning(
        'Cannot calculate wpms because audio len and transcription do not match'
      )
      log.warning(audio_lens)
      log.warning(transcription_lens)
      return None

  def audio_metrics(self):
    ret = {}

    audio_lens = [
      # else assumes .wav case - librosa requires numpy 1.23.1 not 1.24, bugs out
      MP3(f).info.length if f[-4:] == '.mp3' else librosa.get_duration(
        filename=f) for f in self.audio_history
    ]

    ret['avg_wpm'] = self.calc_avg_wpm(audio_lens)

    fillers = [
      get_num_fillers(audio_path) for audio_path in self.audio_history
    ]

    ret['total_fillers'] = sum(fillers)

    silent_time = [
      get_silent_time(audio_path) for audio_path in self.audio_history
    ]

    if len(audio_lens) > 0:
      ret['percent_silent'] = (sum(silent_time) / sum(audio_lens)) * 100

    else:
      ret['percent_silent'] = None
      log.warning('Could not calculate % silent, division by 0! No audio_lens')

    log.debug('Got audio metrics')
    return ret

  def score_conversation(self, do_audio=False):
    messages = EVALUATE_PROMPT.format_prompt(
      transcript=self.history_for_prompt(), text="").to_messages()
    output = chat(messages)

    log.info("Conversation Score Output: %s", output)

    try:
      self.score = json.loads(output.content)
      if do_audio:
        self.score['audio_metrics'] = self.audio_metrics()
    except:
      log.error(f'could not parse json output from prompt: {output.content}')
      # Use fallback mock score
      self.score = {"clarity": 7, "confidence": 6, "vocabulary": 4, "poise": 5}
    return self.score
