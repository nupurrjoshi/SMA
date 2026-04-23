
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Install wordcloud if not installed
#!pip install wordcloud

from wordcloud import WordCloud




# Replace with your file name if different
df = pd.read_csv('youtube1.csv')



print("First 5 Rows:  Untitled1:19 - EDA.py:19")
print(df.head())

print("\nDataset Info:  Untitled1:22 - EDA.py:22")
print(df.info())

print("\nMissing Values:  Untitled1:25 - EDA.py:25")
print(df.isnull().sum())



cols_to_drop = ['href', 'upload', 'YouTube Links', 'likes', 'replies']
existing_cols = [col for col in cols_to_drop if col in df.columns]
df = df.drop(columns=existing_cols)

# Ensure 'name' and 'comment' columns exist before processing
if 'name' in df.columns:
    df['name'] = df['name'].astype(str).str.replace('@', '', regex=False)

if 'comment' in df.columns:
    df['comment'] = df['comment'].astype(str).str.replace('[^a-zA-Z ]', '', regex=True)

print("\nCleaned Data:  Untitled1:41 - EDA.py:41")
print(df.head())


print("\nStatistical Summary:  Untitled1:45 - EDA.py:45")
print(df.describe(include='all  Untitled1:46 - EDA.py:46'))


# ================================
# 📊 VISUALIZATIONS
# ================================

# 1. BAR CHART - Top Users
if 'name' in df.columns:
    df['name'].value_counts().head(10).plot(kind='bar')
    plt.title("Top Users by Number of Comments")
    plt.xlabel("Username")
    plt.ylabel("Count")
    plt.show()


# 2. PIE CHART - User Distribution
if 'name' in df.columns:
    df['name'].value_counts().head(5).plot(kind='pie', autopct='%1.1f%%')
    plt.title("Top 5 Users Distribution")
    plt.ylabel("")
    plt.show()


# 3. HEATMAP - Correlation (Only if numeric columns exist)
numeric_df = df.select_dtypes(include=['int64', 'float64'])

if not numeric_df.empty:
    sns.heatmap(numeric_df.corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.show()



if 'comment' in df.columns:
    text = " ".join(df['comment'].dropna())
    if text:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        plt.imshow(wordcloud)
        plt.axis('off')
        plt.title("Word Cloud of Comments")
        plt.show()



from textblob import TextBlob

# 5. SENTIMENT ANALYSIS
if 'comment' in df.columns:
    df['sentiment'] = df['comment'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    print("\nSentiment Analysis:  Untitled1:98 - EDA.py:98")
    print(df[['comment  Untitled1:99 - EDA.py:99', 'sentiment']].head())

df.to_csv("cleaned_youtube_data.csv", index=False)

print("\n✅ Cleaned dataset saved as 'cleaned_youtube_data.csv'  Untitled1:103 - EDA.py:103")