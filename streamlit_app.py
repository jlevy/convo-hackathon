import logging
from config import api_setup
import streamlit as st
import base64
from audio_recorder_streamlit import audio_recorder
from prompts import SCENARIOS, ConversationState
from speech_handling import synth_text_as_file, transcribe_speech, save_audio
from mutagen.mp3 import MP3

log = logging.getLogger(__name__)
api_setup()

st.set_page_config(layout="wide")


def find_scenario_by_name(name):
  for scenario in SCENARIOS:
    if scenario.name == name:
      return scenario
  raise ValueError(f"Invalid scenario {name}")


def current_scenario():
  return find_scenario_by_name(st.session_state.scenario)


def load_scenario():
  scenario_name = current_scenario()
  if scenario_name:
    st.session_state.convo_state = ConversationState(st.session_state.scenario)
  st.session_state.scenario_details = ""
  st.session_state.is_prepared = False
  st.session_state.is_scorable = False


def start_conversation():
  log.info("Starting conversation")
  load_scenario()
  st.session_state.is_prepared = True
  # TODO: Set job type and history from scenario-defined input fields
  st.session_state.convo_state.greet_user(context="")


def next_step():
  if "step" not in st.session_state:
    st.session_state.step = 1
  else:
    st.session_state.step += 1


with st.sidebar:
  st.markdown(
    "## Conversation Coach",
    unsafe_allow_html=True,
  )

  st.markdown(
    "Who will you have a conversation with?",
    unsafe_allow_html=True,
  )
  st.selectbox("Scenario", [s.name for s in SCENARIOS],
               key="scenario",
               on_change=load_scenario)

  if current_scenario():
    st.markdown(current_scenario().description)

  st.button(label="Start Over", on_click=load_scenario)


def speak(text):
  agent_utterance_audio_file = synth_text_as_file(text)
  with open(agent_utterance_audio_file, 'rb') as audio_file:
    audio_bytes = audio_file.read()
    log.info("Got synth speech size %s: %s", len(audio_bytes),
             agent_utterance_audio_file)
  # https://github.com/streamlit/streamlit/issues/2446
  st.audio(audio_bytes, format='audio/mp3')
  # TODO: Put this into static and serve to make faster:
  #st.write('<audio autoplay src="data:audio/mp3;base64,%s" type="audio/mp3">' %
  #         (base64.b64encode(audio_bytes).decode('utf-8')),
  #         unsafe_allow_html=True)


#def scenario_details_callback(details):
#  log(f'Set scenario details: {details}')
#  st.session_state.scenario_details = details

tab_converse, tab_reflect = st.tabs(["Converse", "Reflect"])


def scenario_details():
  if "scenario_details" in st.session_state:
    return st.session_state.scenario_details
  else:
    return ""


with tab_converse:
  if "is_prepared" not in st.session_state or not st.session_state.is_prepared:
    st.markdown("This part is optional but helps customize the conversaton.")

    st.markdown("Can you say anything else about who you'll be talking with?")

    st.text_input(
      "Details:",
      key="scenario_details",
      disabled=False,
      placeholder=
      "A recruiter I talked to at Google who wants me to call back with details."
    )

    st.button("Let's start the conversation!", on_click=start_conversation)
  else:
    if "convo_state" in st.session_state and st.session_state.convo_state:
      convo_state = st.session_state.convo_state
      if convo_state.active:
        for user_input, ai_response in convo_state.history:
          if user_input:
            st.info(user_input)
          st.warning(ai_response)
        if convo_state.last_ai_response:
          text = convo_state.last_ai_response
          convo_state.last_ai_response = None
          speak(text)

        audio_bytes = audio_recorder(text="",
                                     recording_color="#e8b62c",
                                     neutral_color="#6aa36f",
                                     icon_name="user",
                                     icon_size="2x")
        # TODO: Visual feedback with progress box.
        if audio_bytes:
          log.info("Got audio %s bytes", len(audio_bytes))
          audio_file = save_audio(audio_bytes)
          log.info("Saved audio file: %s", audio_file)
          transcription = transcribe_speech(audio_file)
          if transcription:
            log.info("Got transcription: %s", transcription)
            # Get reply to transcribed audio

            did_update = st.session_state.convo_state.handle_message(
              user_input=transcription,
              audio_file=audio_file,
              context=scenario_details())
            if did_update:
              st.experimental_rerun()


def do_score_callback():
  st.session_state.is_scorable = True


with tab_reflect:
  if "convo_state" in st.session_state:
    convo_state = st.session_state.convo_state
    if convo_state.active:
      score = st.session_state.convo_state.score_conversation(
        do_audio=st.session_state.is_scorable)
      log.info("score %s", score)
      if score:
        st.markdown(f"""
        <div style="background: #eee; padding: 8px;">
        
        🌤️ **Clarity:** **{score["clarity"]}**/10
        
        💪 **Confidence:** **{score["confidence"]}**/10
        
        🧠 **Vocabulary:** **{score["vocabulary"]}**/10
        
        🧘 **Poise:** **{score["poise"]}**/10
        </div>
        """,
                    unsafe_allow_html=True)
      if score and 'audio_metrics' in score:
        audio_metrics = score['audio_metrics']
        log.info("audio metrics: %s", audio_metrics)
        st.markdown(f"""
        <div style="background: #eef; padding: 8px;">
         
        **Total fillers:** {score["total_fillers"]}
        
        **Average WPM:** {score["avg_wpm"]}

        **Percent silent:** {score["percent_silent"]}

        </div>
        """)
      else:
        st.markdown("")
        st.button("Advanced Analysis", on_click=do_score_callback)
