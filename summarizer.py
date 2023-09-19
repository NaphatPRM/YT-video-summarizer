#importing dependencies

import re
from youtube_transcript_api import YouTubeTranscriptApi
import torch
import torchaudio
import textwrap
from transformers import pipeline, AutoTokenizer
import streamlit as st


# Initialize the session state variables
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""

st.set_page_config(page_title="YouTube Video Summarizer", page_icon="🇵🇰", layout="wide")

st.title("YouTube Video Summarizer")

#Enter YT link
youtube_url = st.text_input("Enter the YouTube video URL here (works best for videos under 30 minutes)")

if st.button("Generate Transcript"):
    with st.spinner("Generating transcript"):
        # Extract the video ID from the URL using regular expressions
        match = re.search(r"v=([A-Za-z0-9_-]+)", youtube_url)
        if match:
            video_id = match.group(1)
        else:
            st.error("Invalid YouTube URL")

        # Get the transcript from YouTube API
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Concatenate the transcript into a single string
        transcript_text = ""
        for segment in transcript:
            transcript_text += segment["text"] + " "

        # Store the transcript in the session state
        st.session_state.transcript = transcript_text

        st.write(transcript_text)   # displays transcipt text !


if st.button("Summarize"):
    with st.spinner("Generating Summary"):
        # Retrieve the transcript from the session state
        transcript_text = st.session_state.transcript

        # Instantiate the tokenizer and the summarization pipeline
        tokenizer = AutoTokenizer.from_pretrained('stevhliu/my_awesome_billsum_model')
        summarizer = pipeline("summarization", model='stevhliu/my_awesome_billsum_model', tokenizer=tokenizer)

        # Define chunk size in number of words
        chunk_size = 200 
        # Split the text into chunks
        words = transcript_text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            # Summarize the chunk
            summary = summarizer(chunk, max_length=100, min_length=50, do_sample=False)

            # Extract the summary text
            summary_text = summary[0]['summary_text']

            # Add the summary to our list of summaries
            summaries.append(summary_text)

        # Join the summaries back together into a single summary
        final_summary = ' '.join(summaries)

        st.write(final_summary)
