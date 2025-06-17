import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

data = pd.read_csv('colorsmp.csv')

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Color Name']).toarray()
y = data[['R', 'G', 'B']].values

model = LinearRegression()
model.fit(X, y)

def predict_color_rgb(color_name):
    try:
        input_vector = vectorizer.transform([color_name.lower()]).toarray()
        rgb = model.predict(input_vector).astype(int).flatten()
        print(f"Predicted RGB for {color_name}: {list(rgb)}")
    except ValueError:
        print("Invalid color name. Please try again!")

while True:
    user_input = input("Enter a color name (or 'exit' to stop): ").strip()
    if user_input.lower() == 'exit':
        break
    predict_color_rgb(user_input)
