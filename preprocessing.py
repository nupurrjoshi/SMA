import pandas as pd

# Load dataset
df = pd.read_csv('youtube1.csv')

# View data
print(df.head(20))

# Info about dataset
print(df.info())

# Drop unwanted columns safely
cols_to_drop = ['upload', 'YouTube Links', 'likes', 'replies'] 
existing_cols = [col for col in cols_to_drop if col in df.columns]

df = df.drop(columns=existing_cols)

# Clean first comment text
text = df['comment'].iloc[0]
print("Original:  Untitled1:20 - preprocessing.py:20", text)

text = text.replace('!', '')
print("Cleaned:  Untitled1:23 - preprocessing.py:23", text)

# Remove '@' from usernames
df['name'] = df['name'].str.replace('@', '', regex=False)

# Final output
print(df.head(20))