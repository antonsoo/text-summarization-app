import streamlit as st
from transformers import pipeline

# Set the title and a brief description
st.title("Text Summarization App")
st.write("Enter some text and I'll generate a summary for you!")

# Load the summarization pipeline
@st.cache_resource
def load_summarizer():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

summarizer = load_summarizer()

# Text input area
text_input = st.text_area("Paste your text here:", height=200)

# Summarization button
if st.button("Generate Summary"):
    if text_input:
        # Generate summary
        summary = summarizer(text_input, max_length=130, min_length=30, do_sample=False)

        # Display the summary
        st.subheader("Summary:")
        st.write(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text to summarize.")
