import streamlit as st
import nltk
from text_summarizer import summarize_text, get_text_stats, get_tldr

st.title("Text Summarizer")


# Input text area
text_input = st.text_area(
    "Type or paste your text here",
    height=200,
    placeholder="Enter your text here..."
)

# Action buttons
col1, col2 = st.columns(2)
summary_button = col1.button("Generate Summary", type="primary")
tldr_button = col2.button("TL;DR", type="primary")

if text_input:
    if tldr_button:
        try:
            tldr_text, score = get_tldr(text_input)
            st.markdown("### TL;DR Summary")
            st.markdown(f"<div style='margin-bottom: 1em;'><span style='background-color: #0e4c92; color: white; padding: 0.2em 0.5em; border-radius: 3px;'>Score: {score:.2f}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='padding: 1.5em; border-radius: 8px; background: linear-gradient(135deg, #1e88e5 0%, #0d47a1 100%); color: white; font-size: 1.1em;'>{tldr_text}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-top: 1em; font-size: 0.9em; color: #666;'>This sentence represents {score*100:.0f}% of your text's main ideas based on TF-IDF scoring using NLTK and scikit-learn.</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred while generating TL;DR: {str(e)}")
    
    if summary_button:
        try:
            summary, highlighted_text = summarize_text(text_input)
            stats = get_text_stats(text_input)
            
            st.subheader("3-Sentence Summary")
            st.write(summary)
            
            st.subheader("Text Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sentences", stats["sentences"])
            with col2:
                st.metric("Total Words", stats["words"])
            with col3:
                st.metric("Avg. Sentence Length", f"{stats['avg_sentence_length']:.1f}")
            
            st.subheader("Original Text")
            st.write("(Summary sentences are highlighted in yellow)")
            st.markdown(highlighted_text, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred while processing the text: {str(e)}")
else:
    if summary_button or tldr_button:
        st.warning("Please enter text first.")
