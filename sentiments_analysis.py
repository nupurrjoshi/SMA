

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from pytrends.request import TrendReq

df = pd.read_csv('tweets.csv')

print(df.info())
print(df.head())

def get_polarity(text):
    return TextBlob(str(text)).sentiment.polarity

df['polarity'] = df['Tweet'].apply(get_polarity)

def get_sentiment(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"

df['sentiment'] = df['polarity'].apply(get_sentiment)

df = df.drop(columns=[col for col in ['Unnamed: 0', 'Unnamed: 0.1'] if col in df.columns])

print(df.head())

df.to_csv('sentiment_tweets.csv', index=False)

plt.figure()
sns.countplot(x='sentiment', data=df)
plt.title("Sentiment Count")
plt.show()

plt.figure()
df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title("Sentiment Distribution")
plt.ylabel("")
plt.show()

pos_tweets = df[df['sentiment'] == 'Positive']
text = " ".join(pos_tweets['Tweet'].astype(str))

plt.figure()
wordcloud = WordCloud(width=800, height=400).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Positive Tweets WordCloud")
plt.show()

neg_tweets = df[df['sentiment'] == 'Negative']
text = " ".join(neg_tweets['Tweet'].astype(str))

plt.figure()
wordcloud = WordCloud(width=800, height=400).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Negative Tweets WordCloud")
plt.show()

neu_tweets = df[df['sentiment'] == 'Neutral']
text = " ".join(neu_tweets['Tweet'].astype(str))

plt.figure()
wordcloud = WordCloud(width=800, height=400).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Neutral Tweets WordCloud")
plt.show()

pytrends = TrendReq(hl='en-IN', tz=330)

keywords = ["Data Science", "AI", "Machine Learning"]

pytrends.build_payload(keywords, timeframe='today 12-m', geo='IN')

trends_data = pytrends.interest_over_time()

print(trends_data.head())

plt.figure()
trends_data[keywords].plot()
plt.title("Google Trends Analysis (India)")
plt.xlabel("Date")
plt.ylabel("Search Interest")
plt.show()

related_queries = pytrends.related_queries()

print(related_queries)