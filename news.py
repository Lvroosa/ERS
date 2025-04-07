import streamlit as st
from google import genai
import pandas as pd
import re
import datetime
from streamlit_tags import st_tags
import json
import plotly.express as px
import requests



# Store API keys securely (Replace with your actual API keys)
NEWS_API_KEY = st.secrets["all_my_api_keys"]["NEWS_API_KEY"]
GEMINI_API_KEY = st.secrets["all_my_api_keys"]["GEMINI_API_KEY"]




# Configure Gemini API
client = genai.Client(api_key=GEMINI_API_KEY)
# Streamlit UI
st.title("Tulane University: Sentiment Analysis from News")








# Make it so someone can type in their own keywords to customize the search
search = st_tags(label="Enter your values (press Enter to separate keywords):",
                 text="Add a new value...",
                 value=["Tulane"],  # Default values
                 suggestions=["Tulane University"],  # Optional suggestions
                 key="1")
# Separate the search terms with a plus sign
start_date = st.date_input("Start Date", value= datetime.date.today() - datetime.timedelta(days = 7))
end_date = st.date_input("End Date", value=datetime.date.today())








sports = st.checkbox("Include sports news")
use_cache = st.checkbox("Use cache (uncheck for debugging purposes)", value=True)








@st.cache_data(show_spinner=False, persist=True)
def fetch_news(search, start_date, end_date, sports):
    if sports:
        news_url = (
            f"https://newsapi.org/v2/everything?q={search}&"
            f"from={start_date}&to={end_date}&sortBy=popularity&apiKey={NEWS_API_KEY}"
        )
    else:
        news_url = (
            f"https://newsapi.org/v2/everything?q={search} NOT sports NOT Football&"
            f"from={start_date}&to={end_date}&sortBy=popularity&apiKey={NEWS_API_KEY}"
        )
    response = requests.get(news_url)
    if response.status_code == 200:
        news_data = response.json()
        return news_data.get("articles", [])  # Return the articles
    else:
        return []








@st.cache_data(show_spinner=False, persist=True)
def analyze_sentiment(text_to_analyze, search, sports):
    if sports:
        sentiment_prompt = (
            "Analyze the sentiment of the following news articles in relation to the keywords: "
            f"'{search}'.\n"
            "Assume all articles affect Tulane's reputation positively, neutrally, or negatively. \n"
            "Then, consider how the keywords also get discussed or portrayed in the article.\n"
            "Provide an overall sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, and 1 is very positive(This is a continuous range)) \n"
            "Provide a summary of the sentiment and key reasons why the sentiment is positive, neutral, or negative, "
            "specifically in relation to the keywords.\n"
            "Make sure that you include the score from -1 to 1 in a continuous range (with decimal places) and include the title, "
            "sentiment score, summary, and a statement explaining how the article relates to the keywords.\n"
            "Separate article info by double newlines and always include 'Title:' before the headline and 'Sentiment:' before the score.\n"
            "Only judge the sentiment for each article in terms of how it mentions the keywords. Max amount of titles should be 100.\n\n"
            "If the article merely mentions a quote from a Tulane student, faculty, or staff, mention that on the response.\n"
            f"{text_to_analyze}"
        )
    else:
        sentiment_prompt = (
            "Analyze the sentiment of the following news articles in relation to the keywords: "
            f"'{search}'.\n"
            "Assume all articles affect Tulane's reputation positively, neutrally, or negatively. \n"
            "Then, consider how the keywords also get discussed or portrayed in the article.\n"
            "Provide an overall sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, and 1 is very positive(This is a continuous range)) \n"
            "Provide a summary of the sentiment and key reasons why the sentiment is positive, neutral, or negative, "
            "specifically in relation to the keywords.\n"
            "Make sure that you include the score from -1 to 1 in a continuous range (with decimal places) and include the title, "
            "sentiment score, summary, and a statement explaining how the article relates to the keywords.\n"
            "Separate article info by double newlines and always include 'Title:' before the headline and 'Sentiment:' before the score.\n"
            "If you encounter any articles related to sports, please exclude them from the analysis. Sports articles do not need to be summarized. \n"
            "Only judge the sentiment for each article in terms of how it mentions the keywords. Max amount of titles should be 100.\n\n"
            "If the article merely mentions a quote from a Tulane student, faculty, or staff, mention that on the summary.\n"
            f"{text_to_analyze}"
        )
    gemini_response = client.models.generate_content(model="gemini-1.5-pro-latest", contents=[sentiment_prompt])
    return gemini_response.text if gemini_response and gemini_response.text else ""






if "slider_value" not in st.session_state:
    st.session_state.slider_value = (-1.0, 1.0)
if st.button('Search') or "slider_shown" in st.session_state:
    search = '+'.join(search)
    if use_cache:
        articles = fetch_news(search, start_date, end_date, sports)
    else:
        fetch_news.clear()
        articles = fetch_news(search, start_date, end_date, sports)
   
    if not articles:
        st.write("No articles found.")
    else:
        text_to_analyze = "\n\n".join(
            [f"Title: {article['title']}\nDescription: {article['description']}\nContent: {article['content']}\nURL:{article['url']}" for article in articles]
        )
        if use_cache:
            gemini_response_text = analyze_sentiment(text_to_analyze, search, sports)
        else:
            analyze_sentiment.clear()
            gemini_response_text = analyze_sentiment(text_to_analyze, search, sports)


        def text_to_dataframe(text, articles):
            titles = []
            scores = []
            urls = []
            sections = re.split(r'Title: ', text)[1:]
            for section in sections:
                title_match = re.match(r'(.*?)\nSentiment:', section, re.DOTALL)
                if title_match:
                    title = title_match.group(1).strip()
                    titles.append(title)
                    # Find the corresponding URL for the title
                    url = next((article['url'] for article in articles if article['title'] == title), None)
                    urls.append(url)
                else:
                    titles.append(None)
                    urls.append(None)
                score_match = re.search(r'Sentiment:\s*(-?\d+\.?\d*)', section, re.DOTALL)
                if score_match:
                    scores.append(float(score_match.group(1)))
                else:
                    scores.append(None)
            df = pd.DataFrame({'Title': titles, 'Sentiment': scores, 'URL': urls})
            return df


        df = text_to_dataframe(gemini_response_text, articles)
        sentiment_counts = df['Sentiment'].value_counts()
       
        st.header("Sentiment Score Summary")
        st.write("")
        # Plot sentiment score summary
        st.bar_chart(sentiment_counts)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Sentiment Score", round(df['Sentiment'].mean(), 2))
        with col2:
            st.metric("Number of News Stories", len(df))








        if df['Sentiment'].mean() >= 0.1:
            st.write("Overall sentiment is positive.")
        elif df['Sentiment'].mean() <= -0.1:
            st.write("Overall sentiment is negative.")
        else:
            st.write("Overall sentiment is neutral.")
       
        st.write("---")
        st.header("News Stories")




        st.session_state.slider_shown = True
        st.session_state.slider_value = st.slider("Sentiment Filter", -1.0, 1.0, (-1.0, 1.0), 0.1,)
     
        st.write("")
        sections = gemini_response_text.split("\n\n")
        filtered_sections = []
        for section in sections:
            sentiment_match = re.search(r'Sentiment:\s*(-?\d+\.?\d*)', section)
            if sentiment_match:
                sentiment = float(sentiment_match.group(1))
            if st.session_state.slider_value[0] <= sentiment <= st.session_state.slider_value[1]:
                filtered_sections.append(section)


        for section in filtered_sections:
            title_match = re.search(r'Title:\s*(.*)', section)
            sentiment_match = re.search(r'Sentiment:\s*(-?\d+\.?\d*)', section)
            summary_match = re.search(r'Summary:\s*(.*)', section)
            if title_match and sentiment_match and summary_match:
                title = title_match.group(1)
                sentiment = sentiment_match.group(1)
                summary = summary_match.group(1)
            for article in articles:
                if article['title'] == title:
                    url = article['url']
            st.markdown(f"###  **[{title}]({url})**")
            st.markdown(f"🔹 **Sentiment Score:** `{sentiment}`")
            st.markdown(f" **Summary:** {summary}")
            st.write("---")




     
        st.header("Dataframe of Results")
        st.dataframe(df)