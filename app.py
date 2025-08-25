from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from chat import get_response

app = Flask(__name__)

# Data loading and preprocessing
data = pd.read_csv("cleaned_updated_recipes.csv")

# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['ingredients_list'])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_ingredients)

def recommend_recipes(input_ingredients, selected_region):
    # Transform input ingredients
    input_ingredients_transformed = vectorizer.transform([input_ingredients])
    distances, indices = knn.kneighbors(input_ingredients_transformed)
    
    # Filter recommendations by region if a specific region is selected
    recommendations = data.iloc[indices[0]]
    if selected_region != "All":
        recommendations = recommendations[recommendations['Region'] == selected_region]
    
    return recommendations[['recipe_name', 'ingredients_list', 'Description', 'Procedure', 'Region', 'nutritional_value', 'image_url']].head(5).to_dict(orient='records')

def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route("/predict", methods=['POST'])
def predict():
    text=request.get_json().get("message")
    response=get_response(text)
    message={"answer": response}
    return jsonify(message)
    
    
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ingredients = request.form['ingredients']
        region = request.form['region']
        recommendations = recommend_recipes(ingredients, region)
        return render_template('index.html', recommendations=recommendations, truncate=truncate)
    return render_template('index.html', recommendations=[])

if __name__ == '__main__':
    app.run(debug=True)
