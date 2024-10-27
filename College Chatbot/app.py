from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load your CSV data
data = pd.read_csv(r'C:\Users\omkar\OneDrive\Documents\Flask Project\College Chatbot\question_answers.csv')  # Make sure this file exists
questions = data['question'].tolist()
answers = data['answer'].tolist()

# Create an instance of TfidfVectorizer
vectorizer = TfidfVectorizer()
# Fit the model on the questions
vectors = vectorizer.fit_transform(questions).toarray()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json['question']
    user_vector = vectorizer.transform([user_input]).toarray()
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(user_vector, vectors)
    closest_index = np.argmax(cosine_similarities)
    
    # Find the best match
    best_answer = answers[closest_index]
    return jsonify(answer=best_answer)

if __name__ == '__main__':
    app.run(debug=True)
