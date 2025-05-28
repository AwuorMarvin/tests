import streamlit as st
import PyPDF2
import docx
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Function to extract text from Word document
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading Word document: {e}")
        return ""

# Function to process uploaded file based on type
def process_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(('.docx', '.doc')):
        return extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a PDF or Word document.")
        return ""

# Function for letter frequency distribution
def letter_frequency_analysis(text):
    letters = [char.lower() for char in text if char.isalpha()]
    letter_counts = Counter(letters)
    df = pd.DataFrame.from_dict(letter_counts, orient='index', columns=['Count']).sort_index()
    return df

# Function for paragraph word count analysis
def paragraph_word_count_analysis(text):
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    word_counts = [len(re.findall(r'\b\w+\b', para)) for para in paragraphs]
    return paragraphs, word_counts

# Function for number distribution analysis (single digits 0-9)
def number_distribution_analysis(text):
    numbers = [char for char in text if char.isdigit()]
    number_counts = Counter(numbers)
    all_digits = {str(i): 0 for i in range(10)}
    all_digits.update(number_counts)
    df = pd.DataFrame.from_dict(all_digits, orient='index', columns=['Count']).sort_index()
    return df

# Function for special character analysis
def special_character_analysis(text):
    special_chars = [char for char in text if not char.isalnum() and not char.isspace()]
    char_counts = Counter(special_chars)
    df = pd.DataFrame.from_dict(char_counts, orient='index', columns=['Count']).sort_index()
    return df

# Streamlit app
def main():
    st.title("Document Analysis App")
    st.write("Upload a PDF or Word document to analyze its content.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'doc'])

    if uploaded_file is not None:
        # Extract text
        text = process_file(uploaded_file)
        
        if text:
            st.subheader("Extracted Text Preview")
            st.text_area("Text", text[:500] + "..." if len(text) > 500 else text, height=200)

            # Letter Frequency Distribution
            st.subheader("Letter Frequency Distribution")
            letter_df = letter_frequency_analysis(text)
            if not letter_df.empty:
                fig, ax = plt.subplots()
                sns.barplot(x=letter_df.index, y=letter_df['Count'], ax=ax)
                plt.title("Letter Frequency Distribution")
                plt.xlabel("Letter")
                plt.ylabel("Count")
                st.pyplot(fig)
            else:
                st.write("No letters found in the document.")

            # Paragraph Word Count Analysis
            st.subheader("Paragraph Word Count Analysis")
            paragraphs, word_counts = paragraph_word_count_analysis(text)
            if word_counts:
                df = pd.DataFrame({'Paragraph': [f"Para {i+1}" for i in range(len(paragraphs))], 'Word Count': word_counts})
                st.dataframe(df)
                
                # Paragraph Length Visualization (Histogram)
                st.subheader("Paragraph Length Distribution")
                if word_counts:
                    # Define bins (1-10, 11-20, 21-30, etc.)
                    max_count = max(word_counts, default=10)
                    bin_edges = np.arange(0, max_count + 11, 10)  # Bins: 0-10, 11-20, etc.
                    fig, ax = plt.subplots()
                    sns.histplot(word_counts, bins=bin_edges, kde=False, ax=ax)
                    plt.title("Paragraph Length Distribution (Word Counts)")
                    plt.xlabel("Word Count Range")
                    plt.ylabel("Number of Paragraphs")
                    # Customize x-axis labels to show ranges
                    bin_labels = [f"{int(bin_edges[i])}–{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
                    ax.set_xticks(bin_edges[:-1])  # Set ticks at left edge of bins
                    ax.set_xticklabels(bin_labels, rotation=45)
                    st.pyplot(fig)
                else:
                    st.write("No paragraphs found in the document.")
            else:
                st.write("No paragraphs found in the document.")
            # Paragraph Word Count Analysis
            st.subheader("Paragraph Word Count Analysis")
            paragraphs, word_counts = paragraph_word_count_analysis(text)
            # --- Sentiment Analysis Section ---
            from textblob import TextBlob

            st.subheader("Sentiment Analysis per Paragraph")

            if paragraphs:
                sentiment_data = []
                sentiment_labels = []

                for i, para in enumerate(paragraphs):
                    blob = TextBlob(para)
                    polarity = blob.sentiment.polarity
                    if polarity > 0.1:
                        sentiment = "Positive"
                    elif polarity < -0.1:
                        sentiment = "Negative"
                    else:
                        sentiment = "Neutral"
                    sentiment_data.append({
                        "Paragraph": f"Para {i+1}",
                        "Polarity Score": polarity,
                        "Sentiment": sentiment
                    })
                    sentiment_labels.append(sentiment)

                sentiment_df = pd.DataFrame(sentiment_data)
                st.dataframe(sentiment_df)

                # --- Sentiment Distribution Pie Chart ---
                st.subheader("Sentiment Distribution Overview")
                sentiment_counts = pd.Series(sentiment_labels).value_counts().sort_index()
                fig, ax = plt.subplots()
                ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                    colors=['#66bb6a', '#ff7043', '#ffee58'])  # green, red, yellow
                ax.set_title("Overall Sentiment Distribution")
                st.pyplot(fig)
            else:
                st.info("No paragraphs found for sentiment analysis.")




            # Number Distribution Analysis
            st.subheader("Number Distribution Analysis (Digits 0-9)")
            number_df = number_distribution_analysis(text)
            if not number_df.empty:
                fig, ax = plt.subplots()
                sns.barplot(x=number_df.index, y=number_df['Count'], ax=ax)
                plt.title("Digit Distribution (0-9)")
                plt.xlabel("Digit")
                plt.ylabel("Count")
                plt.xticks(rotation=0)
                st.pyplot(fig)
            else:
                st.write("No digits found in the document.")

            # Special Character Analysis
            st.subheader("Special Character Analysis")
            special_df = special_character_analysis(text)
            if not special_df.empty:
                fig, ax = plt.subplots()
                sns.barplot(x=special_df.index, y=special_df['Count'], ax=ax)
                plt.title("Special Character Distribution")
                plt.xlabel("Character")
                plt.ylabel("Count")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.write("No special characters found in the document.")

if __name__ == "__main__":
    main()