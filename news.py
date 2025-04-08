import streamlit as st
import requests
from google import genai
import pandas as pd
import re
import datetime
from streamlit_tags import st_tags
import json
import plotly.express as px
from newspaper import article
from google.genai.errors import ClientError
import random
import time
import asyncio
from lxml.html.clean import Cleaner


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
            f"https://newsapi.org/v2/everything?q={search} NOT sports NOT Football NOT basketball&"
            f"from={start_date}&to={end_date}&sortBy=popularity&apiKey={NEWS_API_KEY}"
        )
    response = requests.get(news_url)
    if response.status_code == 200:
        news_data = response.json()
        return news_data.get("articles", [])  # Return the articles
    else:
        return []




@st.cache_data(show_spinner=False, persist=True)
def fetch_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return None




@st.cache_data(show_spinner=False, persist=True)
def get_articles_with_full_content(articles):
    """Replace truncated content with full article text."""
    updated_articles = []
    seen_titles = set()
   
    for article in articles:
        title = article["title"]
        if title in seen_titles:
            continue  # Skip duplicate
        seen_titles.add(title)
   
        full_text = fetch_content(article["url"]) if "[+" in article["content"] else article["content"]
        updated_articles.append({
            "title": article["title"],
            "description": article.get("description", "No description available."),
            "content": full_text if full_text else article["content"],
            "url": article["url"]
        })
    return updated_articles


@st.cache_data(show_spinner=False, persist=True)
def format_articles_for_prompt(articles):
    """Format the articles in a way that can be sent to Gemini."""
    return "\n\n".join(
        [f"Title: {article['title']}\nDescription: {article['description']}\nContent: {article['content']}\nURL: {article['url']}"
         for article in articles])




@st.cache_data(show_spinner=False, persist=True)
def analyze_sentiment(text_to_analyze, search, sports, retries=5):
    for attempt in range(retries):
        try:
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
                    "If an article title has already been seen before, do not analyze it again.\n"
                    "If Tulane is mentioned anywhere in the article — even in passing or in an author affiliation — state that clearly in your summary.\n"
                    "Tulane was found in the text. Here is the full content for analysis.\n"
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
                    "If an article title has already been seen before, do not analyze it again.\n"
                    "If Tulane is mentioned anywhere in the article — even in passing or in an author affiliation — state that clearly in your summary.\n"
                    "Tulane was found in the text. Here is the full content for analysis.\n"
                    f"{text_to_analyze}"
                )
            gemini_response = client.models.generate_content(model="gemini-2.5-pro-exp-03-25", contents=[sentiment_prompt])
            return gemini_response.text if gemini_response and gemini_response.text else ""
        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60  # Default wait time (1 minute)
                retry_delay_match = re.search(r"'retryDelay': '(\d+)s'", str(e))
                if retry_delay_match:
                    wait_time = int(retry_delay_match.group(1))  # Use API's recommended delay
               
                print(f"⚠️ API quota exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"❌ API request failed: {e}")
                return "❌ API error encountered."
        except requests.exceptions.ConnectionError:
            wait_time = 2 ** attempt + random.uniform(0, 1)
            print(f"Connection error. Waiting for {wait_time:.2f} seconds before retrying...")
            time.sleep(wait_time)
        return "❌ API failed after multiple attempts."


semaphore = asyncio.Semaphore(5)


async def limited_process(batch_df, search, batch_size, total_batches):
    async with semaphore:
        return await asyncio.to_thread(process_batch, batch_df, search, batch_size, total_batches)


async def analyze_in_batches_concurrent(articles, search, sports, batch_size=10):
    all_responses = []


    total_batches = len(articles) // batch_size + (1 if len(articles) % batch_size != 0 else 0)
    print(f"Total batches: {total_batches}")


    batch_tasks = []
    for i in range(0, len(articles), batch_size):
        batch_df = articles[i:i + batch_size]


        task = limited_process(batch_df, search, i // batch_size + 1, total_batches)
        batch_tasks.append(task)
   
    all_responses = await asyncio.gather(*batch_tasks)
    return '\n\n'.join(all_responses)


def run_async_batches(articles, search, sports, batch_size = 10):
    return asyncio.run(analyze_in_batches_concurrent(articles, search, sports, batch_size))


def process_batch(batch_df, search, i, total_batches):


    processed_articles = get_articles_with_full_content(batch_df)
    formatted_batch = format_articles_for_prompt(processed_articles)
    print(f"Processing batch {i} of {total_batches}...")
    response = analyze_sentiment(formatted_batch, search, sports)
    return response












if "slider_value" not in st.session_state:
    st.session_state.slider_value = (-1.0, 1.0)
if st.button('Search') or "slider_shown" in st.session_state:
    search = '+'.join(search)
    if use_cache:
        articles = fetch_news(search, start_date, end_date, sports)
        unique_articles = []
        seen_titles = set()


        for article in articles:
            if article['title'] not in seen_titles:
                unique_articles.append(article)
                seen_titles.add(article['title'])
        articles = unique_articles
    else:
        fetch_news.clear()
        articles = fetch_news(search, start_date, end_date, sports)
        unique_articles = []
        seen_titles = set()


        for article in articles:
            if article['title'] not in seen_titles:
                unique_articles.append(article)
                seen_titles.add(article['title'])


        articles = unique_articles
   
    if not articles:
        st.write("No articles found.")
    else:
        if use_cache:
            gemini_response_text = run_async_batches(articles, search, sports, batch_size=10)
        else:
            gemini_response_text = run_async_batches(articles, search, sports, batch_size=10)
           




        def text_to_dataframe(text, articles):
            rows = []
            sections = re.split(r'Title:\s*', text)[1:]


            for section in sections:
                title_match = re.match(r'(.*?)\nSentiment:', section, re.DOTALL)
                sentiment_match = re.search(r'Sentiment:\s*(-?\d+\.?\d*)', section)


                if title_match and sentiment_match:
                    title = title_match.group(1).strip()
                    sentiment = float(sentiment_match.group(1))
                    url = next((article['url'] for article in articles if article['title'].strip() == title), None)
                    rows.append({
                        'Title': title,
                        'Sentiment': sentiment,
                        'URL': url
                    })


            return pd.DataFrame(rows)




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
        filtered_df = df[(df['Sentiment'] >= st.session_state.slider_value[0]) &
                 (df['Sentiment'] <= st.session_state.slider_value[1])]


        for _, row in filtered_df.iterrows():
            st.markdown(f"###  **[{row['Title']}]({row['URL']})**")
            st.markdown(f"🔹 **Sentiment Score:** `{row['Sentiment']}`")
           
            # Grab summary from the original text using the title
            pattern = rf"Title:\s*{re.escape(row['Title'])}\s*.*?Sentiment:\s*-?\d+\.?\d*\s*Summary:\s*(.*?)(?:\n|$)"
            match = re.search(pattern, gemini_response_text, re.DOTALL)
            if match:
                summary = match.group(1).strip()
                st.markdown(f"**Summary:** {summary}")
            else:
                st.markdown("⚠️ Summary not found.")
            st.write("---")

        st.header("Dataframe of Results")
        st.dataframe(df)