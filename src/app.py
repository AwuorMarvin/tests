import streamlit as st
import PyPDF2
import docx
import re
from collections import Counter
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt



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
    if uploaded_file.name.lower().endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.lower().endswith(('.docx', '.doc')):
        return extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a PDF or Word document.")
        return ""

# Function for letter frequency distribution
def letter_frequency_analysis(text):
    lower_counts = Counter(char for char in text if char.islower())
    upper_counts = Counter(char for char in text if char.isupper())
    all_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    data = {
        'Letter': all_letters,
        'Lowercase': [lower_counts.get(letter, 0) for letter in all_letters],
        'Uppercase': [upper_counts.get(letter.upper(), 0) for letter in all_letters]
    }
    df = pd.DataFrame(data)
    return df

# Function for paragraph word count analysis
def paragraph_word_count_analysis(text):
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    word_counts = [len(re.findall(r'\b\w+\b', para)) for para in paragraphs]
    return paragraphs, word_counts

# Function for number distribution analysis (single digits 0-9)
def number_distribution_analysis(text):
    digits = re.findall(r'\b[0-9]\b', text)
    digit_counts = Counter(digits)
    all_digits = {str(i): digit_counts.get(str(i), 0) for i in range(10)}
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

    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'doc'])

    if uploaded_file is not None:
        text = process_file(uploaded_file)
        
        if text:
            st.subheader("Extracted Text Preview")
            st.text_area("Text", text[:500] + "..." if len(text) > 500 else text, height=200)

            # --- Document Summary ---
            st.subheader("Document Summary")
            # Calculate metrics
            words = re.findall(r'\b[a-zA-Z]+\b', text)  # Only alphabetic words, exclude numbers
            total_words = len(words)
            paragraphs, word_counts = paragraph_word_count_analysis(text)
            total_paragraphs = len(paragraphs)
            total_characters = len(text)
            avg_reading_speed = 200  # words per minute (average adult)
            reading_time = total_words / avg_reading_speed
            reading_time = max(1, int(round(reading_time)))  # At least 1 minute

            # Display metrics as cards
            card1, card2, card3, card4 = st.columns(4, gap="large")
            card1.metric("Total Words", f"{total_words:,}")
            card2.metric("Total Paragraphs", f"{total_paragraphs:,}")
            card3.metric("Total Characters", f"{total_characters:,}")
            card4.metric("Est. Reading Time (min)", f"{reading_time}")

            # --- Paragraph Word Count Analysis & Sentiment Analysis per Paragraph ---
            st.subheader("Paragraph Analysis")
            col1, col2 = st.columns(2, gap="large")

            with col1:
               
                paragraphs, word_counts = paragraph_word_count_analysis(text)
                if word_counts:
                    df = pd.DataFrame({'Paragraph': [f"Para {i+1}" for i in range(len(paragraphs))], 'Word Count': word_counts})
                    st.dataframe(df)
                    # Paragraph Length Visualization (Histogram) using Altair
                    import altair as alt
                    hist_df = pd.DataFrame({'Word Count': word_counts})
                    hist_chart = alt.Chart(hist_df).mark_bar().encode(
                        alt.X('Word Count:Q', bin=alt.Bin(maxbins=20)),
                        y='count()'
                    ).properties(title="Paragraph Length Distribution (Word Counts)", width=400, height=300)
                    st.altair_chart(hist_chart, use_container_width=False)
                else:
                    st.write("No paragraphs found in the document.")

            with col2:
                
                from textblob import TextBlob
                if 'paragraphs' not in locals():
                    paragraphs, _ = paragraph_word_count_analysis(text)
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
                else:
                    st.info("No paragraphs found for sentiment analysis.")

            # --- Letter Frequency Distribution & Special Character Distribution ---
            st.subheader("Character Analysis")
            col3, col4 = st.columns(2, gap="large")

            with col3:
                letter_df = letter_frequency_analysis(text)
                if not letter_df.empty:
                    # Prepare data for grouped bar chart: one x-axis, two bars per letter
                    letter_chart = pd.melt(
                        letter_df,
                        id_vars=['Letter'],
                        value_vars=['Lowercase', 'Uppercase'],
                        var_name='Case',
                        value_name='Count'
                    )
                    import altair as alt
                    chart = alt.Chart(letter_chart).mark_bar().encode(
                        x=alt.X('Letter:N', title='Letter'),
                        y=alt.Y('Count:Q', title='Count'),
                        color=alt.Color('Case:N', scale=alt.Scale(domain=['Lowercase', 'Uppercase'], range=['blue', 'red'])),
                        tooltip=['Letter', 'Case', 'Count']
                    ).properties(
                        title="Letter Frequency Distribution (Lowercase vs Uppercase)",
                        width=400,
                        height=300
                    )
                    st.altair_chart(chart, use_container_width=False)
                else:
                    st.write("No letters found in the document.")

            with col4:
               
                special_df = special_character_analysis(text)
                if not special_df.empty:
                    special_df = special_df.reset_index().rename(columns={'index': 'Character'})
                    chart = px.bar(special_df, x='Character', y='Count', title="Special Character Distribution", width=400, height=300)
                    st.plotly_chart(chart, use_container_width=False)
                else:
                    st.write("No special characters found in the document.")

            # --- Number Distribution Analysis & Sentiment Distribution Donut Chart ---
            st.subheader("Number & Sentiment Overview")
            col5, col6 = st.columns(2, gap="large")

            with col5:
                
                number_df = number_distribution_analysis(text)
                if not number_df.empty:
                    number_df = number_df.reset_index().rename(columns={'index': 'Digit'})
                    chart = px.bar(number_df, x='Digit', y='Count', title="Digit Distribution (0-9)", width=400, height=300)
                    st.plotly_chart(chart, use_container_width=False)
                else:
                    st.write("No digits found in the document.")

            with col6:
                
                if 'sentiment_labels' not in locals():
                    # If not already calculated, do it now
                    paragraphs, _ = paragraph_word_count_analysis(text)
                    from textblob import TextBlob
                    sentiment_labels = []
                    for para in paragraphs:
                        blob = TextBlob(para)
                        polarity = blob.sentiment.polarity
                        if polarity > 0.1:
                            sentiment = "Positive"
                        elif polarity < -0.1:
                            sentiment = "Negative"
                        else:
                            sentiment = "Neutral"
                        sentiment_labels.append(sentiment)
                if sentiment_labels:
                    sentiment_counts = pd.Series(sentiment_labels).value_counts().sort_index()
                    fig = px.pie(
                        names=sentiment_counts.index,
                        values=sentiment_counts.values,
                        title="Overall Sentiment Distribution",
                        hole=0.5,  # This makes it a donut chart
                        color_discrete_sequence=[ '#ffee58','#66bb6a', '#ff7043',]
                    )
                    st.plotly_chart(fig, use_container_width=False)
                else:
                    st.info("No paragraphs found for sentiment analysis.")

            # --- Word Cloud and Top Words Table ---
            st.subheader("Word Cloud & Top Words")
            wc_col, tbl_col = st.columns(2, gap="large")

            # Tokenize and count word frequencies (exclude numbers)
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            word_counts = Counter(words)
            most_common_words = word_counts.most_common(20)

            # Slider to select number of words in word cloud/table
            num_words = st.slider("Number of words to display", min_value=5, max_value=20, value=10, step=1)

            # Prepare data for word cloud and table
            top_words = dict(most_common_words[:num_words])

            with wc_col:
                st.markdown("**Word Cloud**")
                if top_words:
                    wc = WordCloud(width=400, height=300, background_color='white', colormap='tab20',
                                   max_words=num_words).generate_from_frequencies(top_words)
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.write("Not enough words for word cloud.")

            with tbl_col:
                st.markdown("**Top Words Table**")
                if top_words:
                    df_top_words = pd.DataFrame(list(top_words.items()), columns=['Word', 'Frequency'])
                    st.dataframe(df_top_words, use_container_width=True)
                else:
                    st.write("Not enough words for table.")

if __name__ == "__main__":
    main()