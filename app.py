import streamlit as st 
import sounddevice as sd
from scipy.io.wavfile import write
import os
import threading
from faster_whisper import WhisperModel
import pyttsx3
from groq import Groq
import uuid
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import textwrap
from gtts import gTTS
import pygame
from io import BytesIO
import subprocess
import tempfile
import os
# ========== FOLDERS ========== 
os.makedirs("transcripts", exist_ok=True)
os.makedirs("feedback", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ========== GLOBALS ========== 
SAMPLE_RATE = 16000
DEFAULT_WHISPER_MODEL_SIZE = "tiny.en"

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    layout="wide", 
    page_title="AI Technical Interviewer",
    page_icon="🤖",
    initial_sidebar_state="collapsed"
)

# ========== CSS STYLING ==========
css = """
<style>
    /* Base Styles */
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
        animation: fadeIn 1s ease-in-out;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        animation: slideUp 0.5s ease-out;
    }
    
    .card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        transform: translateY(-5px);
    }
    
    .question-card {
        background-color: #F0F9FF;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 5px;
        margin-bottom: 1.2rem;
        animation: pulseBlue 2s infinite;
    }
    
    .feedback-card {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 1.2rem;
        border-radius: 5px;
        margin-bottom: 1.2rem;
    }
    
    .score-display {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #1E40AF;
        padding: 1rem;
        animation: scaleIn 0.5s ease-out;
    }
    
    .btn-primary {
        background-color: #2563EB;
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        transition: all 0.3s;
        border: none;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
        display: inline-block;
        cursor: pointer;
        width: 100%;
    }
    
    .btn-primary:hover {
        background-color: #1D4ED8;
        box-shadow: 0 6px 10px rgba(29, 78, 216, 0.3);
        transform: translateY(-2px);
    }
    
    .btn-success {
        background-color: #10B981;
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        transition: all 0.3s;
        border: none;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
        display: inline-block;
        cursor: pointer;
        width: 100%;
    }
    
    .btn-success:hover {
        background-color: #059669;
        box-shadow: 0 6px 10px rgba(5, 150, 105, 0.3);
        transform: translateY(-2px);
    }
    
    .btn-warning {
        background-color: #F59E0B;
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        transition: all 0.3s;
        border: none;
        box-shadow: 0 4px 6px rgba(245, 158, 11, 0.2);
        display: inline-block;
        cursor: pointer;
        width: 100%;
    }
    
    .btn-warning:hover {
        background-color: #D97706;
        box-shadow: 0 6px 10px rgba(217, 119, 6, 0.3);
        transform: translateY(-2px);
    }
    
    .btn-danger {
        background-color: #EF4444;
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        transition: all 0.3s;
        border: none;
        box-shadow: 0 4px 6px rgba(239, 68, 68, 0.2);
        display: inline-block;
        cursor: pointer;
        width: 100%;
    }
    
    .btn-danger:hover {
        background-color: #DC2626;
        box-shadow: 0 6px 10px rgba(220, 38, 38, 0.3);
        transform: translateY(-2px);
    }
    
    .recording-pulse {
        display: inline-block;
        width: 18px;
        height: 18px;
        background-color: #EF4444;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
        margin-right: 10px;
    }
    
    .domain-icon {
        font-size: 2.2rem;
        margin-right: 0.5rem;
        color: #3B82F6;
    }
    
    .centered-content {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 70vh;
        text-align: center;
    }
    
    .welcome-animation {
        animation: bounceIn 1s;
    }
    
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 0.875rem;
        text-align: center;
        margin-right: 0.5rem;
    }
    
    .badge-blue {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    
    .badge-green {
        background-color: #D1FAE5;
        color: #065F46;
    }
    
    .badge-yellow {
        background-color: #FEF3C7;
        color: #92400E;
    }
    
    .badge-purple {
        background-color: #EDE9FE;
        color: #5B21B6;
    }
    
    .badge-red {
        background-color: #FEE2E2;
        color: #B91C1C;
    }
    
    .intro-box {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem 3rem;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    .stImageContainer{
    max-width: 30%;
    width: 786.4px;
    position: relative;
    left: 500px
    }
    
    .intro-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, #3B82F6, #10B981, #F59E0B);
    }
    
    .feature-list {
        list-style-type: none;
        padding-left: 0.5rem;
        margin-top: 1.5rem;
    }
    
    .feature-list li {
        margin-bottom: 1rem;
        padding-left: 2rem;
        position: relative;
    }
    
    .feature-list li::before {
        content: "✓";
        position: absolute;
        left: 0;
        color: #10B981;
        font-weight: bold;
    }
    
    .logo-container {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .logo {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: contain;
        padding: 0.5rem;
        background: white;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        animation: float 3s ease-in-out infinite;
    }
    
   .reward-card {
    position: relative;
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: white;
    border-radius: 15px;
    padding: 2rem;
    text-align: center;
    overflow: hidden;
    box-shadow: 0 10px 20px rgba(79, 70, 229, 0.3);
    margin-bottom: 2rem;
}
    
    .reward-badge {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .reward-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
   .confetti-container {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    pointer-events: none;
    z-index: 1;
}

    .confetti {
    position: absolute;
    border-radius: 50%;
    animation: confetti 5s ease-in-out infinite;
}

    
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
        height: 10px;
        border-radius: 10px;
    }
    
    /* Animations */
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
        }
        
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 10px rgba(239, 68, 68, 0);
        }
        
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes scaleIn {
        from { transform: scale(0.8); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    @keyframes bounceIn {
        0% { transform: scale(0.3); opacity: 0; }
        50% { transform: scale(1.05); opacity: 1; }
        70% { transform: scale(0.9); }
        100% { transform: scale(1); }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    @keyframes pulseBlue {
        0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.2); }
        70% { box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
        100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 10px rgba(79, 70, 229, 0.5); }
        to { box-shadow: 0 0 20px rgba(79, 70, 229, 0.8); }
    }
    
   @keyframes confetti {
    0% { 
        transform: translateY(0) rotate(0); 
        opacity: 1; 
    }
    100% { 
        transform: translateY(1000px) rotate(360deg); 
        opacity: 0; 
    }
}

    
    /* Modal styling */
    .modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    }
    
    .modal-content {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        width: 90%;
        max-width: 600px;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.25);
        position: relative;
    }
    
    .close-button {
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 1.5rem;
        cursor: pointer;
        background: none;
        border: none;
        color: #6B7280;
    }
    
    .close-button:hover {
        color: #1F2937;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F1F5F9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94A3B8;
    }
    
    /* For the robot logo animation */
    .robot-logo {
        animation: robotBounce 2s infinite alternate;
    }
    
    @keyframes robotBounce {
        0% { transform: translateY(0); }
        100% { transform: translateY(-10px); }
    }
    
    /* Trophy styles for rewards */
    .trophy {
        font-size: 5rem;
        margin-bottom: 1rem;
        display: inline-block;
        animation: trophyGlow 2s infinite alternate;
    }
    
    @keyframes trophyGlow {
        0% { text-shadow: 0 0 10px gold; }
        100% { text-shadow: 0 0 25px gold, 0 0 40px orange; }
    }
    
    /* Certificate styles */
    .certificate {
        background: linear-gradient(135deg, #fff, #f9f9f9);
        border: 15px solid #3B82F6;
        border-radius: 15px;
        padding: 3rem;
        position: relative;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .certificate::before {
        content: "";
        position: absolute;
        top: 1px;
        left: 1px;
        right: 1px;
        bottom: 1px;
        border: 2px solid #3B82F6;
        border-radius: 10px;
    }
    
    .certificate-title {
        font-size: 2.5rem;
        color: #1E40AF;
        margin-bottom: 1.5rem;
        font-family: 'Times New Roman', serif;
    }
    
    .certificate-text {
        font-size: 1.2rem;
        margin-bottom: 2rem;
        color: #1F2937;
    }
    
    .certificate-name {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 2rem;
        text-decoration: underline;
    }
    
    .certificate-date {
        font-size: 1rem;
        color: #6B7280;
        margin-top: 3rem;
    }
    
    .certificate-seal {
        display: inline-block;
        font-size: 4rem;
        margin: 1rem;
        color: #3B82F6;
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# ========== LOGO AND ANIMATIONS ==========
def get_robot_logo_html():
    robot_logo = """
    <div class="logo-container">
        <div class="robot-logo">
            <svg width="120" height="120" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="120" height="120" rx="60" fill="#EFF6FF"/>
                <rect x="30" y="30" width="60" height="60" rx="15" fill="#3B82F6"/>
                <circle cx="45" cy="50" r="5" fill="white"/>
                <circle cx="75" cy="50" r="5" fill="white"/>
                <rect x="45" y="70" width="30" height="5" rx="2.5" fill="white"/>
                <rect x="25" y="25" width="10" height="20" rx="5" fill="#F59E0B"/>
                <rect x="85" y="25" width="10" height="20" rx="5" fill="#F59E0B"/>
            </svg>
        </div>
    </div>
    """
    return robot_logo

# ========== UTILITY FUNCTIONS ==========
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



@st.cache_resource
def load_whisper():
    return WhisperModel(DEFAULT_WHISPER_MODEL_SIZE, compute_type="int8")

whisper = load_whisper()

# ========== SESSION STATE ========== 
default_state = {
    "current_question": "",
    "question_number": 0,
    "scores": [],
    "domain": "Python",
    "finished": False,
    "answered": False,
    "interview_started": False,
    "spoken": False,
    "recording_started": False,
    "audio_recording_filename": "temp.wav",
    "audio_thread": None,
    "recording_start_time": None,
    "transcript": "",
    "feedback": "",
    "score": 0,
    "page": "intro",  # intro -> setup -> interview -> result
    "interview_in_progress": False,
    "recording_duration": 0,  # No fixed limit - will be controlled by user
    "interview_ended": False,
    "username": "",
    "questions_asked": 0,
    "total_score": 0,
    # Code-related variables
    "is_coding_question": False,
    "code_submission": "",
    "code_language": "python",
    "code_result": None,
    "code_feedback": "",
    "supported_languages": ["python", "java", "javascript", "cpp", "html", "css"],  # Supported languages
    "read_question_aloud": False  # New variable to control auto-reading
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ========== CORE FUNCTIONS ==========
def speak_sync(text, slow=False, rate=2):
    """
    Function to convert English text to speech with Indian accent/voice
    
    Parameters:
        text (str): The English text to convert to speech
        slow (bool): Whether to use the slow TTS option (defaults to False for normal speed)
        rate (float): Playback rate modifier (1.0 is normal, >1.0 is faster, <1.0 is slower)
                     Only works if pygame mixer supports it
    """
    try:
        # Use English language with standard TLD for more natural speed
        # Set slow=False to ensure normal speed base
        tts = gTTS(text=text, lang='en', tld='com', slow=False)
        
        # Save to a BytesIO object (memory) instead of a file
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        # Initialize pygame mixer
        pygame.mixer.init(frequency=24000)  # Higher frequency for faster playback
        
        # Load the audio file and play it
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
    except Exception as e:
        print(f"TTS Error: {str(e)}")

def speak_question():
    """
    Function to speak the current question using text-to-speech
    Only triggered when the speak aloud button is clicked
    """
    if st.session_state.current_question:
        # Reset spoken flag to False so we can speak again if needed
        st.session_state.spoken = False
        
        # Create a thread to speak the question so it doesn't block the UI
        speech_thread = threading.Thread(
            target=speak_sync,
            args=(st.session_state.current_question,)
        )
        speech_thread.start()
        st.session_state.spoken = True
def on_speak_button_click():
    """
    Function to handle the speak aloud button click
    """
    # Set the flag to indicate the user wants to hear the question
    st.session_state.read_question_aloud = True
    # Call the speak function directly
    speak_question()

def _record(filename):
    """
    Record audio to a file until manually stopped
    """
    try:
        # Create a new recording stream
        recording = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        recording.start()
        start_time = time.time()
        
        frames = []
        # Use a shared variable for control instead of accessing session_state directly
        recording_active = True
        
        # Continue recording while the flag is True
        while recording_active:
            audio, _ = recording.read(SAMPLE_RATE)
            if len(audio) > 0:
                frames.append(audio.copy())
            
            # Check if we should stop recording by checking a file-based signal
            # or by checking a global variable instead of session_state
            if hasattr(threading.current_thread(), "stop_requested") and threading.current_thread().stop_requested:
                recording_active = False
            
            time.sleep(0.1)
        
        # Stop and close the recording when done
        recording.stop()
        recording.close()
        
        # Save the recorded audio data
        if len(frames) > 0:
            audio_data = np.concatenate(frames, axis=0)
            write(filename, SAMPLE_RATE, audio_data)
            return True
        return False
    except Exception as e:
        # Don't use st.error here since we're in a thread
        print(f"Recording error: {str(e)}")
        return False

def start_recording():
    """
    Start the audio recording in a separate thread
    """
    # Reset state and prepare filename
    st.session_state.recording_started = True
    st.session_state.recording_start_time = time.time()
    st.session_state.audio_recording_filename = f"temp_q{st.session_state.question_number + 1}.wav"
    
    # Start recording in a thread
    thread = threading.Thread(
        target=_record,
        args=(st.session_state.audio_recording_filename,)
    )
    thread.daemon = True  # Make thread terminate when main program exits
    thread.stop_requested = False  # Custom attribute to communicate with the thread
    thread.start()
    
    # Store the thread object in session state
    st.session_state.audio_thread = thread

def stop_recording():
    """
    Stop the audio recording and process the result
    """
    # Signal the thread to stop recording
    if st.session_state.audio_thread and hasattr(st.session_state.audio_thread, "stop_requested"):
        st.session_state.audio_thread.stop_requested = True
    
    # Update the session state
    st.session_state.recording_started = False
    
    # Wait for the recording thread to finish
    if st.session_state.audio_thread and st.session_state.audio_thread.is_alive():
        st.session_state.audio_thread.join(timeout=2.0)  # Add timeout to prevent hanging
    
    try:
        # Check if the file exists and has content
        if not os.path.exists(st.session_state.audio_recording_filename) or os.path.getsize(st.session_state.audio_recording_filename) == 0:
            st.warning("⚠ No audio was recorded. Please try again.")
            return
        
        # Process the recording
        with st.spinner("Transcribing your answer..."):
            transcript = transcribe_audio(st.session_state.audio_recording_filename).strip()
            st.session_state.transcript = transcript
            
            if not transcript:
                st.warning("⚠ No voice detected or transcription failed. Please try again.")
                return
            
            # Evaluate answer
            with st.spinner("Evaluating your answer..."):
                feedback = evaluate_answer(st.session_state.current_question, transcript)
                score = extract_score(feedback)
                
                st.session_state.feedback = feedback
                st.session_state.score = score
                st.session_state.scores.append(score)
                st.session_state.total_score += score
                st.session_state.questions_asked += 1
                st.session_state.answered = True  # Set this to True for all questions
                st.session_state.finished = True  # Set this to True for all questions
            
            # Save logs
            with open(f"transcripts/q{st.session_state.question_number + 1}.txt", "w") as f:
                f.write(f"Q: {st.session_state.current_question}\nA: {transcript}")
            with open(f"feedback/q{st.session_state.question_number + 1}_feedback.txt", "w") as f:
                f.write(feedback)
            with open("logs/session_log.txt", "a") as log:
                avg_score = sum(st.session_state.scores) / len(st.session_state.scores)
                session_id = uuid.uuid4()
                log.write(f"Session ID: {session_id} | User: {st.session_state.username} | "
                         f"Domain: {st.session_state.domain} | "
                         f"Q{st.session_state.question_number + 1} Score: {score} | Avg: {avg_score:.2f}\n")
            
    except Exception as e:
        st.error(f"❌ Error processing recording: {str(e)}")
        st.session_state.recording_started = False

def transcribe_audio(filename):
    segments, _ = whisper.transcribe(filename)
    return " ".join([segment.text for segment in segments])


def generate_question(domain):
    apikey="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"
    client = Groq(
        api_key=apikey,
    )
    
    # Modified prompt to specifically exclude coding questions
    prompt = f"""Generate one challenging technical interview question related to {domain}.
    
    Important requirements:
    - Make the question specific and thoughtful
    - DO NOT generate any coding questions that ask the candidate to write code
    - DO NOT ask the candidate to implement algorithms or data structures
    - DO NOT ask the candidate to write functions, classes, or any code snippets
    - Focus instead on conceptual understanding, design principles, theory, or system knowledge
    - Questions should test understanding rather than coding ability
    
    Generate a theoretical or conceptual question only."""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        stream=False, 
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    
    return response.choices[0].message.content.strip()

def evaluate_answer(question, answer):
    apikey="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"
    client = Groq(
        api_key=apikey,
    )
    
    # First, check if the answer is relevant to the question
    relevance_prompt = f"""Question: {question}
Answer: {answer}

Analyze if this answer is relevant to the question asked. 
Is the answer attempting to address the question that was asked?
Respond with just "RELEVANT" or "NOT_RELEVANT". 

Guidelines:
- If the answer seems to address the question, even if it's partially correct or incorrect, mark as "RELEVANT"
- If the answer is completely off-topic, discussing something unrelated to the question, mark as "NOT_RELEVANT"
- If the answer is very generic with no specific relation to the question topic, mark as "NOT_RELEVANT"
- If the answer is just repeating the question or contains random text, mark as "NOT_RELEVANT"
"""

    relevance_check = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        stream=False,
        messages=[
            {
                "role": "user",
                "content": relevance_prompt,
            }
        ]
    )
    
    relevance_result = relevance_check.choices[0].message.content.strip()
    
    # If the answer is not relevant, return a simple message
    if "NOT_RELEVANT" in relevance_result:
        return "Please give a valid answer that addresses the question. Your response doesn't seem to be related to the question asked."
    
    # If the answer is relevant, proceed with the full evaluation
    # Using a single prompt format for all questions (technical domain questions only)
    prompt = f"""Question: {question}
Answer: {answer}

Evaluate this answer objectively on a scale of 0-10. Provide feedback in the following format:
Score: [0-10]

Strengths:
- [Point 1]
- [Point 2]

Areas for Improvement:
- [Point 1]
- [Point 2]

Overall Feedback: [2-3 sentences with constructive feedback]
"""
    
    feedback = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        stream=False, 
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    
    return feedback.choices[0].message.content.strip()

# ========== NEW FUNCTIONS FOR CODING QUESTIONS ==========
def generate_coding_question(domain):
    """
    Generate a coding question based on the selected domain
    """
    apikey="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"
    client = Groq(
        api_key=apikey,
    )
    
    # Map domain to appropriate language
    language = get_language_from_domain(domain)
    
    prompt = f"""Generate one challenging but practical coding question related to {domain} that uses {language} programming language.
    
    Important requirements:
    - Make the coding question specific, clear, and doable within 10-15 minutes
    - Include a clear problem statement
    - Specify input and expected output formats
    - Make sure it's appropriate for an interview setting
    - Focus on core concepts in {domain}
    - The solution should be implementable in {language}
    - Avoid extremely complex algorithms that would take too long to implement in an interview
    - Do not include the solution in your response
    
    Format your response as:
    
    # Problem Title
    
    ## Problem Statement
    [Clear description of the problem]
    
    ## Input Format
    [Description of input format]
    
    ## Output Format
    [Description of expected output]
    
    ## Example
    Input: [example input]
    Output: [example output]
    
    ## Notes
    [Optional hints or constraints]
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        stream=False, 
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    
    return response.choices[0].message.content.strip()

def update_default_state():
    supported_languages = [
        "python", 
        "java", 
        "javascript", 
        "cpp", 
        "html", 
        "css"
    ]
    
    return supported_languages
def evaluate_code_submission(question, code, language):
    """
    Evaluate the submitted code using Groq API with language-specific criteria
    """
    apikey="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"
    client = Groq(
        api_key=apikey,
    )
    
    # First, check if the code is relevant to the question
    relevance_prompt = f"""Question: {question}
Code Submission ({language}):
{code}

Analyze if this code submission is relevant to the question asked. 
Is the code attempting to solve the problem described in the question?
Respond with just "RELEVANT" or "NOT_RELEVANT".

Guidelines:
- If the code seems to be attempting to solve the problem, even if it's incorrect, mark as "RELEVANT"
- If the code is completely off-topic with no relation to the problem, mark as "NOT_RELEVANT"
- If the code is very generic with no specific relation to the problem, mark as "NOT_RELEVANT"
- If the submission is just random text or code with no apparent purpose, mark as "NOT_RELEVANT"
"""

    relevance_check = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        stream=False,
        messages=[
            {
                "role": "user",
                "content": relevance_prompt,
            }
        ]
    )
    
    relevance_result = relevance_check.choices[0].message.content.strip()
    
    # If the code is not relevant, return a simple message
    if "NOT_RELEVANT" in relevance_result:
        return "Please provide code that addresses the problem. Your submission doesn't seem to be related to the question asked."
    
    # Create language-specific guidance for evaluation
    language_guidance = ""
    if language == "python":
        language_guidance = """
- Check if the solution follows Python conventions (PEP 8)
- Check for Pythonic idioms and features like list comprehensions, generators when appropriate
- Evaluate efficiency in terms of time and space complexity
- Check for proper exception handling if applicable
"""
    elif language == "java":
        language_guidance = """
- Check if the solution follows Java conventions and naming standards
- Check for proper OOP principles and design
- Evaluate efficient use of Java collections and native APIs
- Check for proper exception handling if applicable
"""
    elif language == "javascript" or language == "js":
        language_guidance = """
- Check if the solution follows JavaScript best practices
- Assess modern JS features usage (ES6+) when appropriate
- Evaluate asynchronous code handling if applicable
- Check for proper error handling if applicable
"""
    elif language == "cpp" or language == "c++":
        language_guidance = """
- Check if the solution follows C++ conventions and best practices
- Evaluate memory management and potential memory leaks
- Check for efficient use of STL and algorithms
- Evaluate performance considerations
"""
    else:
        language_guidance = """
- Check if the solution follows common coding best practices
- Evaluate algorithmic efficiency
- Check for proper error/exception handling if applicable
"""
    
    prompt = f"""Question: {question}

Submitted Code ({language}):
{language}
{code}


Please evaluate this {language} code submission thoroughly on a scale of 0-10. Provide detailed feedback in the following format:

Score: [0-10]

Correctness:
- [Assess if the code correctly solves the problem]
- [Mention any edge cases missed]

Code Quality:
- [Comment on code style, readability]
- [Assess efficiency and performance]
- [Note any best practices followed or missed]

Language-Specific Assessment:
{language_guidance}

Areas for Improvement:
- [Point 1]
- [Point 2]

Solution Approach:
[Briefly explain a correct solution approach]

Overall Feedback: [2-3 sentences with constructive feedback]
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        stream=False, 
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    
    return response.choices[0].message.content.strip()

def run_code(code, language="python"):
    """
    Run the submitted code and return the result based on the programming language
    """
    result = {
        "output": "",
        "error": "",
        "success": False
    }
    
    # Create a temporary file to save the code
    with tempfile.NamedTemporaryFile(suffix=f".{language}", delete=False, mode='w') as temp_file:
        temp_file.write(code)
        temp_filename = temp_file.name
    
    try:
        # Run the code based on the language
        if language == "python":
            # Use subprocess to execute the code
            process = subprocess.run(
                ["python", temp_filename],
                capture_output=True,
                text=True,
                timeout=5  # Set a timeout to prevent infinite loops
            )
            
            if process.returncode == 0:
                result["output"] = process.stdout
                result["success"] = True
            else:
                result["error"] = process.stderr
        
        elif language == "java":
            # For Java, we need to extract the class name, compile and then run
            # Simple class name extraction - this is a basic approach
            class_name = None
            for line in code.split('\n'):
                if 'public class' in line:
                    parts = line.split('public class')[1].strip().split(' ')
                    if parts:
                        class_name = parts[0].strip().split('{')[0].strip()
                        break
            
            if not class_name:
                result["error"] = "Could not identify Java class name. Make sure you have a 'public class ClassName' declaration."
                return result
            
            # Rename temp file to match class name
            java_file = f"{os.path.dirname(temp_filename)}/{class_name}.java"
            os.rename(temp_filename, java_file)
            
            # Compile
            compile_process = subprocess.run(
                ["javac", java_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if compile_process.returncode != 0:
                result["error"] = f"Compilation Error: {compile_process.stderr}"
                return result
            
            # Run
            run_process = subprocess.run(
                ["java", "-cp", os.path.dirname(java_file), class_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if run_process.returncode == 0:
                result["output"] = run_process.stdout
                result["success"] = True
            else:
                result["error"] = f"Runtime Error: {run_process.stderr}"
        
        elif language == "javascript" or language == "js":
            # Run with Node.js
            process = subprocess.run(
                ["node", temp_filename],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if process.returncode == 0:
                result["output"] = process.stdout
                result["success"] = True
            else:
                result["error"] = process.stderr
        
        elif language == "cpp" or language == "c++":
            # Compile C++ code
            compiled_file = f"{os.path.splitext(temp_filename)[0]}"
            compile_process = subprocess.run(
                ["g++", "-std=c++11", temp_filename, "-o", compiled_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if compile_process.returncode != 0:
                result["error"] = f"Compilation Error: {compile_process.stderr}"
                return result
            
            # Run the compiled executable
            run_process = subprocess.run(
                [compiled_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if run_process.returncode == 0:
                result["output"] = run_process.stdout
                result["success"] = True
            else:
                result["error"] = f"Runtime Error: {run_process.stderr}"
                
        elif language == "html":
            # For HTML, there's no execution, so we just return the HTML content
            result["output"] = "HTML code doesn't produce console output. Use a web browser to view."
            result["success"] = True
            
        elif language == "css":
            # Same for CSS
            result["output"] = "CSS code doesn't produce console output. Use a web browser to view with HTML."
            result["success"] = True
            
        else:
            result["error"] = f"Language '{language}' is not supported for execution yet."
    
    except subprocess.TimeoutExpired:
        result["error"] = "Execution timed out. Your code may contain an infinite loop."
    except Exception as e:
        result["error"] = f"An error occurred: {str(e)}"
    
    finally:
        # Clean up the temporary files
        try:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
            # Also clean up java class files if applicable
            if language == "java" and 'class_name' in locals() and class_name:
                class_file = f"{os.path.dirname(temp_filename)}/{class_name}.class"
                if os.path.exists(class_file):
                    os.unlink(class_file)
            # Clean up C++ compiled executable
            if (language == "cpp" or language == "c++") and 'compiled_file' in locals():
                if os.path.exists(compiled_file):
                    os.unlink(compiled_file)
        except:
            pass
    
    return result
def get_language_from_domain(domain):
    """
    Map a domain to an appropriate programming language for code execution
    """
    domain_language_map = {
        "Python": "python",
        "Java": "java",
        "C++": "cpp",
        "JavaScript": "javascript",
        "React": "javascript",
        "Node.js": "javascript",
        "Full Stack": "javascript",  # Default to JavaScript for full stack
        "Data Science": "python",
        "Machine Learning": "python",
        "DevOps": "python",  # Default to Python for DevOps
        "Cloud Computing": "python",  # Default to Python for Cloud
        "Database": "sql",  # SQL might need special handling
        "System Design": "pseudocode",  # Not executable
        "Algorithms": "python"  # Default to Python for algorithms
    }
    
    return domain_language_map.get(domain, "python")  # Default to Python if domain not found


# ========== UPDATES TO SESSION STATE ==========
# Add these keys to your default_state dictionary:
#
# "is_coding_question": False,
# "code_submission": "",
# "code_language": "python",
# "code_result": None,
# "code_feedback": "",
# "supported_languages": ["python"]  # Add more as you implement them

def extract_score(feedback):
    for line in feedback.splitlines():
        if line.lower().startswith("score:"):
            try:
                score_text = line.split(":")[1].strip()
                # Handle cases like "Score: 7/10" or just "Score: 7"
                score_text = score_text.split("/")[0].strip()
                return int(score_text)
            except:
                pass
    
    # Fallback: look for any number between 0 and 10
    for line in feedback.splitlines():
        for token in line.split():
            if token.isdigit() and 0 <= int(token) <= 10:
                return int(token)
    return 5  # Default score if extraction fails

def reset_interview():
    for key in default_state:
        st.session_state[key] = default_state[key]

def get_domain_icon(domain):
    icons = {
        "Python": "🐍",
        "Java": "☕",
        "C++": "⚙",
        "Full Stack": "🌐",
        "Data Science": "📊",
        "DevOps": "🔄",
        "Machine Learning": "🤖",
        "Cloud Computing": "☁",
        "JavaScript": "📱",
        "React": "⚛",
        "Node.js": "📦",
        "Database": "🗄",
        "System Design": "🏗",
        "Algorithms": "🧮"
    }
    return icons.get(domain, "💻")

def create_score_chart():
    if not st.session_state.scores:
        return None
    
    fig, ax = plt.subplots(figsize=(4, 4))
    scores = st.session_state.scores
    avg_score = sum(scores) / len(scores)
    
    # Create a donut chart
    ax.pie([avg_score, 10-avg_score], 
           colors=['#3B82F6', '#E5E7EB'],
           wedgeprops=dict(width=0.4, edgecolor='w'),
           startangle=90)
    
    # Add text in center
    ax.text(0, 0, f"{avg_score:.1f}", ha='center', va='center', fontsize=24, fontweight='bold')
    ax.text(0, -0.2, "Average", ha='center', va='center', fontsize=12)
    
    # Remove axes
    ax.axis('equal')
    plt.axis('off')
    
    return fig

def get_reward_badge(score_percentage):
    if score_percentage >= 90:
        return "🏆", "Expert Interviewer", "Congratulations on your exceptional performance! You've demonstrated expert-level knowledge."
    elif score_percentage >= 80:
        return "🥇", "Advanced Proficiency", "Excellent work! You've shown strong technical knowledge and communication skills."
    elif score_percentage >= 70:
        return "🥈", "Strong Performer", "Great job! You've demonstrated solid understanding of technical concepts."
    elif score_percentage >= 60:
        return "🥉", "Competent Technologist", "Good effort! You've shown competence in technical areas."
    elif score_percentage >= 50:
        return "🎖", "Promising Talent", "Decent work! You're on the right track with your technical knowledge."
    else:
        return "", "Better Luck Next Time", "You've taken the first steps. Keep learning and practicing!"


def get_certificate_html(username, domain, num_questions, avg_score):
    date_str = datetime.now().strftime("%B %d, %Y")
    certificate = f"""
    <div class="certificate">
        <div class="certificate-title">Certificate of Achievement</div>
        <div class="certificate-text">This is to certify that</div>
        <div class="certificate-name">{username}</div>
        <div class="certificate-text">has successfully completed a technical interview assessment in <strong>{domain}</strong>, 
        answering {num_questions} questions with an average score of <strong>{avg_score:.1f}/10</strong></div>
        <div class="certificate-seal">🏅</div>
        <div class="certificate-date">Issued on {date_str}</div>
    </div>
    """
    return certificate

def generate_certificate_image(username, domain, num_questions, avg_score):
    """
    Generate a certificate as a PIL Image
    """
    # Create a white image with appropriate dimensions
    width, height = 800, 600
    certificate = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(certificate)
    
    # Try to load fonts - fallback to default if not available
    try:
        title_font = ImageFont.truetype("arial.ttf", 36)
        name_font = ImageFont.truetype("arial.ttf", 28)
        text_font = ImageFont.truetype("arial.ttf", 18)
        seal_font = ImageFont.truetype("arial.ttf", 72)
    except IOError:
        # Fallback to default font
        title_font = ImageFont.load_default()
        name_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        seal_font = ImageFont.load_default()
    
    # Draw certificate border
    border_width = 20
    draw.rectangle([(border_width, border_width), (width-border_width, height-border_width)], 
                  outline="#3B82F6", width=5)
    
    # Draw title
    title = "Certificate of Achievement"
    title_width = draw.textlength(title, font=title_font)
    draw.text(((width - title_width) / 2, 60), title, fill="#1E40AF", font=title_font)
    
    # Draw text
    draw.text((width/2, 120), "This is to certify that", fill="#1F2937", font=text_font, anchor="mm")
    
    # Draw name
    draw.text((width/2, 180), username, fill="#1E3A8A", font=name_font, anchor="mm")
    
    # Draw certificate text
    cert_text = f"has successfully completed a technical interview assessment in {domain}, " \
               f"answering {num_questions} questions with an average score of {avg_score:.1f}/10"
               
    # Wrap text to fit certificate width
    wrapped_text = textwrap.fill(cert_text, width=40)
    lines = wrapped_text.split('\n')
    y_position = 250
    for line in lines:
        draw.text((width/2, y_position), line, fill="#1F2937", font=text_font, anchor="mm")
        y_position += 30
    
    # Draw date
    from datetime import datetime
    date_str = datetime.now().strftime("%B %d, %Y")
    draw.text((width/2, 500), f"Issued on {date_str}", fill="#6B7280", font=text_font, anchor="mm")
    
    # Add "seal" or emoji - simplified approach without the problematic code
    seal_text = "🏅"
    draw.text((width/2, 400), seal_text, fill="#3B82F6", font=seal_font, anchor="mm")
    
    return certificate
def download_certificate():
    """
    Generate and create a download link for the certificate
    """
    if not st.session_state.username or not st.session_state.scores:
        return None
    
    username = st.session_state.username
    domain = st.session_state.domain
    num_questions = len(st.session_state.scores)
    avg_score = sum(st.session_state.scores) / num_questions if num_questions > 0 else 0
    
    # Generate certificate image
    certificate_img = generate_certificate_image(username, domain, num_questions, avg_score)
    
    # Save to BytesIO
    buf = BytesIO()
    certificate_img.save(buf, format="PNG")
    buf.seek(0)
    
    # Encode to base64 for download
    img_str = base64.b64encode(buf.read()).decode()
    
    # Create download link
    href = f'<a href="data:image/png;base64,{img_str}" download="certificate_{username}.png" class="btn-success" style="text-decoration:none; text-align:center; display:block; padding:10px; margin-top:10px;">Download Certificate</a>'
    
    return href

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

def create_confetti_html():
    """
    Create confetti elements for the celebration animation
    """
    confetti_html = '<div class="confetti-container" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden; pointer-events: none;">'
    
    for i in range(30):
        x = random.randint(0, 100)
        y = random.randint(-100, 0)
        delay = random.uniform(0, 5)
        size = random.randint(5, 15)
        color = random.choice(['#FCD34D', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6'])
        
    return confetti_html
        
# 1. Modify the next_question() function

# 3. Update the next_question() function to reset the spoken flag
def next_question(is_coding=False):
    """
    Generate a new question and reset all question-related states
    """
    # Increment question counter
    st.session_state.question_number += 1
    
    # Set coding question flag
    st.session_state.is_coding_question = is_coding
    
    # Set appropriate language based on domain
    if is_coding:
        st.session_state.code_language = get_language_from_domain(st.session_state.domain)
    
    # Generate new question
    try:
        with st.spinner("Generating next question..."):
            # Generate domain-specific question directly
            if is_coding:
                st.session_state.current_question = generate_coding_question(st.session_state.domain)
            else:
                st.session_state.current_question = generate_question(st.session_state.domain)
    except Exception as e:
        st.error(f"Error generating question: {str(e)}")
        if is_coding:
            st.session_state.current_question = f"Write a function in {st.session_state.domain} to solve the following problem: [Problem description would go here]"
        else:
            st.session_state.current_question = f"Tell me about a challenging problem you solved in {st.session_state.domain}."
    
    # Reset all question-related states
    st.session_state.spoken = False
    st.session_state.answered = False
    st.session_state.finished = False
    st.session_state.recording_started = False
    st.session_state.transcript = ""
    st.session_state.feedback = ""
    st.session_state.score = 0
    st.session_state.code_submission = ""
    st.session_state.code_result = None
    st.session_state.code_feedback = ""



# ========== PAGE RENDERING ==========
def render_intro_page():
    st.markdown(get_robot_logo_html(), unsafe_allow_html=True)
    st.markdown("<h1 class='main-header welcome-animation'>AI Technical Interviewer</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='intro-box'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Prepare for Technical Interviews with AI</h2>", unsafe_allow_html=True)
    st.markdown("<p>Welcome to the AI Technical Interviewer! Practice your technical interview skills with our AI-powered interviewer that asks questions, evaluates your responses, and provides feedback.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <ul class="feature-list">
        <li>Answer questions verbally and get real-time transcription</li>
        <li>Receive detailed feedback and scoring on your responses</li>
        <li>Practice with questions across various technical domains</li>
        <li>Track your progress and improvement over time</li>
        <li>No time limit - answer at your own pace</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("<p>Click the button below to set up your interview session!</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Take Interview", key="start_setup", help="Begin setting up your interview"):
            st.session_state.page = "setup"
            st.rerun()

def render_setup_page():
    st.markdown(get_robot_logo_html(), unsafe_allow_html=True)
    st.markdown("<h1 class='main-header'>Interview Setup</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Customize Your Interview</h2>", unsafe_allow_html=True)
    
    # Add this descriptive text to ensure content is visible
    st.markdown("<p>Please enter your details below to personalize your technical interview experience.</p>", unsafe_allow_html=True)
    
    # Username input
    username = st.text_input("Your Name:", value=st.session_state.username, 
                             placeholder="Enter your name", key="input_username")
    st.session_state.username = username
    
    # Domain selection
    domain_options = ["Python", "Java", "C++", "JavaScript", "React", "Node.js", 
                     "Full Stack", "Data Science", "Machine Learning", "DevOps", 
                     "Cloud Computing", "Database", "System Design", "Algorithms"]
    
    domain = st.selectbox("Technical Domain:", domain_options, 
                         index=domain_options.index(st.session_state.domain) if st.session_state.domain in domain_options else 0,
                         help="Select the technical domain for your interview questions")
    st.session_state.domain = domain
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_disabled = not st.session_state.username.strip()
        if st.button("Start Interview", disabled=start_disabled, key="start_interview", 
                   help="Begin your technical interview"):
            st.session_state.page = "interview"
            st.session_state.interview_started = True
            # Initialize with first domain-specific question
            st.session_state.question_number = 1  # Start at 1 instead of 0
            st.session_state.current_question = ""  # This will trigger generation in render_interview_page
            st.rerun()

# Also need to modify the interview page rendering to generate a question if none exists

def render_interview_page():
    # Header
    domain_icon = get_domain_icon(st.session_state.domain)
    st.markdown(f"<h1 class='main-header'>{domain_icon} {st.session_state.domain} Technical Interview</h1>", unsafe_allow_html=True)
    
    # Question Card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Generate first question if none exists
    if not st.session_state.current_question:
        # For the first question, randomly decide if it's a coding question based on the domain
        is_coding_first = random.choice([True, False])
        
        # Generate domain-specific question directly
        try:
            with st.spinner("Generating question..."):
                if is_coding_first:
                    st.session_state.current_question = generate_coding_question(st.session_state.domain)
                    st.session_state.is_coding_question = True
                else:
                    st.session_state.current_question = generate_question(st.session_state.domain)
                    st.session_state.is_coding_question = False
        except Exception as e:
            st.error(f"Error generating question: {str(e)}")
            st.session_state.current_question = f"Explain a key concept in {st.session_state.domain}."
            st.session_state.is_coding_question = False
    
    # Set appropriate question title based on question number and type
    if st.session_state.is_coding_question:
        question_title = f"Coding Question {st.session_state.question_number}"
    else:
        question_title = f"Question {st.session_state.question_number}"
    st.markdown(f"""
    <div class='question-card'>
        <h3>{question_title}</h3>
        <p>{st.session_state.current_question}</p>
    </div>
    """, unsafe_allow_html=True)

    
    # Replace the existing speak button with your new one
    if not st.session_state.spoken:
        if st.button("🔊 Speak Question Aloud", key="speak_button", on_click=on_speak_button_click): 
            pass
    
    # Speak the question if not already spoken and it's not a coding question
    if not st.session_state.spoken and not st.session_state.is_coding_question:
        with st.spinner("Speaking question..."):
            speak_question()
    
    # The rest of the function remains the same...
    # Different interfaces for coding vs. speaking questions
    if st.session_state.is_coding_question:
        # Coding interface
        if not st.session_state.answered:
            # Code editor
            code_submission = st.text_area("Your Code Solution:", 
                                         value=st.session_state.code_submission,
                                         height=300,
                                         key="code_editor")
            st.session_state.code_submission = code_submission
            
            # Run and Submit buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Run Code", key="run_code"):
                    if not st.session_state.code_submission.strip():
                        st.warning("Please enter code before running.")
                    else:
                        with st.spinner("Running your code..."):
                            result = run_code(st.session_state.code_submission, st.session_state.code_language)
                            st.session_state.code_result = result
            
            with col2:
                if st.button("Submit Solution", key="submit_code"):
                    if not st.session_state.code_submission.strip():
                        st.warning("Please enter code before submitting.")
                    else:
                        with st.spinner("Evaluating your solution..."):
                            feedback = evaluate_code_submission(
                                st.session_state.current_question,
                                st.session_state.code_submission,
                                st.session_state.code_language
                            )
                            score = extract_score(feedback)
                            
                            st.session_state.code_feedback = feedback
                            st.session_state.feedback = feedback
                            st.session_state.score = score
                            st.session_state.scores.append(score)
                            st.session_state.total_score += score
                            st.session_state.questions_asked += 1
                            st.session_state.answered = True
                            st.session_state.finished = True
                            
                            # Save logs
                            with open(f"transcripts/q{st.session_state.question_number}_code.txt", "w") as f:
                                f.write(f"Q: {st.session_state.current_question}\nA: {st.session_state.code_submission}")
                            with open(f"feedback/q{st.session_state.question_number}_feedback.txt", "w") as f:
                                f.write(feedback)
                            
                            st.rerun()
            
            # Show code execution results if available
            if st.session_state.code_result:
                st.markdown("<div class='feedback-card'>", unsafe_allow_html=True)
                st.markdown("<h3>Code Execution Result:</h3>", unsafe_allow_html=True)
                
                if st.session_state.code_result["success"]:
                    st.code(st.session_state.code_result["output"], language="text")
                else:
                    st.error("Error:")
                    st.code(st.session_state.code_result["error"], language="text")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
    else:
        # Theory question interface - Now with text area option
        if not st.session_state.answered:
            # Add text area for typing answers
            text_answer = st.text_area("Type your answer here:", 
                                      height=200,
                                      key="text_answer",
                                      help="Type your answer or use voice recording below")
            
            # Submit text answer button
            if st.button("Submit Answer", key="submit_text_answer"):
                if not text_answer.strip():
                    st.warning("Please type an answer before submitting.")
                else:
                    with st.spinner("Evaluating your answer..."):
                        # Store the text answer in transcript field
                        st.session_state.transcript = text_answer
                        
                        # Evaluate the answer
                        feedback = evaluate_answer(st.session_state.current_question, text_answer)
                        score = extract_score(feedback)
                        
                        # Update session state
                        st.session_state.feedback = feedback
                        st.session_state.score = score
                        st.session_state.scores.append(score)
                        st.session_state.total_score += score
                        st.session_state.questions_asked += 1
                        st.session_state.answered = True
                        st.session_state.finished = True
                        
                        # Save logs
                        with open(f"transcripts/q{st.session_state.question_number + 1}.txt", "w") as f:
                            f.write(f"Q: {st.session_state.current_question}\nA: {text_answer}")
                        with open(f"feedback/q{st.session_state.question_number + 1}_feedback.txt", "w") as f:
                            f.write(feedback)
                        with open("logs/session_log.txt", "a") as log:
                            avg_score = sum(st.session_state.scores) / len(st.session_state.scores)
                            session_id = uuid.uuid4()
                            log.write(f"Session ID: {session_id} | User: {st.session_state.username} | "
                                    f"Domain: {st.session_state.domain} | "
                                    f"Q{st.session_state.question_number + 1} Score: {score} | Avg: {avg_score:.2f}\n")
                        
                        st.rerun()
            
            # Add separator between text and voice options
            st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
            st.markdown("<p><em>Or record your answer using voice:</em></p>", unsafe_allow_html=True)
            
            # Original voice recording controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if not st.session_state.recording_started:
                    if st.button("🎙 Start Recording", key="start_rec", help="Begin recording your answer"):
                        start_recording()
                        st.rerun()
            
            with col2:
                if st.session_state.recording_started:
                    if st.button("⏹ Stop Recording", key="stop_rec", help="Stop recording and evaluate answer"):
                        stop_recording()
                        st.rerun()
    
    # Question navigation buttons
    if st.session_state.answered:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("➡ Next Question", key="next_regular_q", help="Proceed to the next regular question"):
                next_question(is_coding=False)
                st.rerun()
        
        with col2:
            if st.button("💻 Next Coding Question", key="next_coding_q", help="Proceed to the next coding question"):
                next_question(is_coding=True)
                st.rerun()
        
        with col3:
            # Allow ending interview if any questions have been answered
            if st.session_state.questions_asked > 0:
                if st.button("🏁 End Interview", key="end_interview", help="Finish interview and see results"):
                    st.session_state.page = "result"
                    st.session_state.interview_ended = True
                    st.rerun()
    else:
        # Only show the End Interview button if not in the middle of a question
        if not st.session_state.recording_started and st.session_state.questions_asked > 0:
            if st.button("🏁 End Interview", key="end_interview_early", help="Finish interview and see results"):
                st.session_state.page = "result"
                st.session_state.interview_ended = True
                st.rerun()
        
    # Show recording indicator and timer if currently recording
    if st.session_state.recording_started:
        elapsed_time = int(time.time() - st.session_state.recording_start_time)
        minutes, seconds = divmod(elapsed_time, 60)
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-top: 10px;">
            <span class="recording-pulse"></span>
            <span>Recording: {minutes:02d}:{seconds:02d}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Show transcript if available
    if st.session_state.transcript:
        st.markdown("<div class='feedback-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Your Response:</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>{st.session_state.transcript}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show feedback if available
    if st.session_state.feedback:
        st.markdown("<div class='feedback-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Feedback:</h3>", unsafe_allow_html=True)
        
        # Process the feedback for better rendering
        feedback_parts = st.session_state.feedback.splitlines()
        for part in feedback_parts:
            if part.lower().startswith("score:"):
                score_value = part.split(":")[1].strip()
                st.markdown(f"<p><strong>{part}</strong></p>", unsafe_allow_html=True)
            elif part.startswith("Strengths:") or part.startswith("Areas for Improvement:") or part.startswith("Overall Feedback:") or part.startswith("Correctness:") or part.startswith("Code Quality:") or part.startswith("Solution Approach:"):
                st.markdown(f"<p><strong>{part}</strong></p>", unsafe_allow_html=True)
            elif part.startswith("-"):
                st.markdown(f"<p style='margin-left: 20px;'>{part}</p>", unsafe_allow_html=True)
            elif part.strip():
                st.markdown(f"<p>{part}</p>", unsafe_allow_html=True)
                
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
def render_result_page():
    st.markdown(get_robot_logo_html(), unsafe_allow_html=True)
    st.markdown("<h1 class='main-header'>Interview Results</h1>", unsafe_allow_html=True)
    
    if not st.session_state.scores:
        st.warning("No questions were answered. Please complete at least one question.")
        if st.button("Back to Start"):
            reset_interview()
            st.rerun()
        return
    
    # Calculate total score
    total_questions = len(st.session_state.scores)
    avg_score = sum(st.session_state.scores) / total_questions
    score_percentage = (avg_score / 10) * 100
    
    # Get reward based on score
    badge_icon, badge_title, badge_description = get_reward_badge(score_percentage)
    
    # Reward card with animation - but only if we have a badge icon
    if badge_icon:
        st.markdown(f"""
        <div class="reward-card" style="position: relative;">
            {create_confetti_html()}
            <span class="reward-badge">{badge_icon}</span>
            <h2 class="reward-title">{badge_title}</h2>
            <p>{badge_description}</p>
            <p>Questions Answered: {total_questions} | Average Score: {avg_score:.1f}/10</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Alternative version without badge icon
        st.markdown(f"""
        <div class="reward-card" style="position: relative;">
            {create_confetti_html()}
            <h2 class="reward-title">{badge_title}</h2>
            <p>{badge_description}</p>
            <p>Questions Answered: {total_questions} | Average Score: {avg_score:.1f}/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Score breakdown
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Score Breakdown</h2>", unsafe_allow_html=True)
    
    # Display chart
    fig = create_score_chart()
    if fig:
        st.pyplot(fig)
    
    # Question-by-question scores
    for i, score in enumerate(st.session_state.scores):
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="flex: 1;">Question {i+1}</div>
            <div style="flex: 2;">
                <div style="background-color: #E5E7EB; border-radius: 4px; height: 8px; width: 100%;">
                    <div style="background-color: #3B82F6; border-radius: 4px; height: 8px; width: {score*10}%;"></div>
                </div>
            </div>
            <div style="flex: 0 0 30px; text-align: right;">{score}/10</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Certificate
    if st.session_state.username:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Your Certificate</h2>", unsafe_allow_html=True)
        certificate_html = get_certificate_html(
            st.session_state.username, 
            st.session_state.domain,
            total_questions,
            avg_score
        )
        st.markdown(certificate_html, unsafe_allow_html=True)
        
        download_link = download_certificate()
        if download_link:
            st.markdown(download_link, unsafe_allow_html=True)
        
            st.markdown("</div>", unsafe_allow_html=True)
        
     # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start New Interview", key="new_interview"):
            reset_interview()
            st.session_state.page = "setup"
            st.rerun()
    
    with col2:
        if st.button("Back to Home", key="back_home"):
            reset_interview()
            st.rerun()

# ========== MAIN APP FLOW ==========
def main():
    # Store logs for analytics
    log_file = "logs/app_usage.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a") as log:
        log.write(f"{datetime.now()} - App accessed\n")
    
    # Create folders for transcripts and feedback if they don't exist
    os.makedirs("transcripts", exist_ok=True)
    os.makedirs("feedback", exist_ok=True)
    
    # Render appropriate page based on app state
    if st.session_state.page == "intro":
        render_intro_page()
    elif st.session_state.page == "setup":
        render_setup_page()
    elif st.session_state.page == "interview":
        render_interview_page()
    elif st.session_state.page == "result":
        render_result_page()
if __name__ == "__main__":
    main()