#!/usr/bin/env python3
"""
Recipe Recommender System using Word2Vec
Run this script to get recipe recommendations based on your ingredients!
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading required NLTK data...")
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download('omw-1.4', quiet=True)

class RecipeRecommender:
    def __init__(self):
        self.df_clean = None
        self.w2v_model = None
        self.all_recipes_vector = None
        self.all_tokens = None
        
    def standardize_text(self, input_text):
        """Clean and standardize text data"""
        cleaned_text = []

        for text in input_text:
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = text.replace('\n',' ')
            text = text.strip()
            text = re.sub(' +', ' ', text)

            tokens = word_tokenize(text.lower())

            unit_mapping = {
                "tsp": "teaspoon", "tbsp": "tablespoon",
                "oz": "ounce", "g": "gram", "lb": "pound"
            }

            standardized_tokens = []
            for token in tokens:
                if not token.isnumeric():
                    if token in unit_mapping:
                        token = unit_mapping[token]
                    standardized_tokens.append(token)

            lemmatizer = WordNetLemmatizer()
            standardized_tokens = [lemmatizer.lemmatize(token) for token in standardized_tokens]

            stop_words = set(stopwords.words("english"))
            standardized_tokens = [token for token in standardized_tokens if token not in stop_words]

            common_words = ["teaspoon", "tablespoon", "ounce", "gram", "pound", "cup", 
                          "chopped", "fresh", "ground", "large", "sliced", "peeled",
                          "cut", "freshly", "finely", "plus", "white", "clove", "room", 
                          "dry", "inch"]
            standardized_tokens = [token for token in standardized_tokens if token not in common_words]

            standardized_text = " ".join(standardized_tokens)
            cleaned_text.append(standardized_text)

        return cleaned_text

    def remove_patterns(self, list_text):
        """Remove numeric patterns from text"""
        text_wo_pattern_lst = []
        for text in list_text:
            tokens = text.split()
            tokens = [re.sub(r'\d+$', '', token) for token in tokens]
            tokens = [token for token in tokens if not re.match(r'^\d+[A-Za-z]+\d+$', token)]
            filtered_tokens = [token for token in tokens if not re.match(r'\d+(?:\/\d+)?(?:[a-z°°]+)?', token)]
            filtered_text = " ".join(filtered_tokens)
            text_wo_pattern_lst.append(filtered_text)
        return text_wo_pattern_lst

    def load_data(self):
        """Load and preprocess recipe data"""
        print("\n>> Loading recipe data...")
        
        # Load data from JSON files
        file_path1 = "recipes_raw_nosource_ar.json"
        file_path2 = "recipes_raw_nosource_epi.json"
        file_path3 = "recipes_raw_nosource_fn.json"
        
        try:
            allrecipes = pd.read_json(file_path1).transpose()
            epicurious = pd.read_json(file_path2).transpose()
            foodnetwork = pd.read_json(file_path3).transpose()
        except FileNotFoundError as e:
            print(f"[ERROR] Recipe data files not found!")
            print(f"Please download the required JSON files from:")
            print(f"https://eightportions.com/datasets/Recipes/")
            raise e
        
        df = pd.concat([allrecipes, epicurious, foodnetwork])
        
        print(f"[OK] Loaded {len(df)} recipes")
        
        # Preprocess data
        print(">> Cleaning data...")
        df = df.reset_index()
        df.drop(columns=['index', 'picture_link'], inplace=True)
        df.dropna(inplace=True)
        
        # Convert ingredients to set
        df['ingredients'] = df['ingredients'].apply(set)
        df['ingredient_item'] = df['ingredients'].str.len()
        
        # Create items column
        df['items'] = df['ingredients'].apply(lambda x: ', '.join(x))
        
        # Drop duplicates
        df.drop(columns='ingredients', inplace=True)
        df.drop_duplicates(keep='first', inplace=True)
        df = df.reset_index(drop=True)
        
        self.df_clean = df
        print(f"[OK] Cleaned data: {len(df)} unique recipes")
        
        return df

    def train_model(self):
        """Train Word2Vec model on recipe data"""
        print("\n>> Training Word2Vec model (this may take a few minutes)...")
        
        # Combine all text
        all_text = self.df_clean['title'] + ' ' + self.df_clean['items'] + ' ' + self.df_clean['instructions']
        
        # Standardize text
        cleaned_ver = self.standardize_text(all_text)
        
        # Remove patterns
        clean_text_lst = self.remove_patterns(cleaned_ver)
        
        # Tokenize
        all_tokens = []
        for text in clean_text_lst:
            all_tokens.append(word_tokenize(text))
        
        self.all_tokens = all_tokens
        
        # Train Word2Vec model
        self.w2v_model = Word2Vec(sentences=all_tokens, vector_size=100, 
                                  window=5, min_count=1, workers=4)
        
        print("[OK] Model trained successfully!")
        
        # Pre-compute all recipe vectors
        print(">> Computing recipe vectors...")
        all_recipes_vector = []
        
        for tokens in all_tokens:
            recipe_vector = [0] * self.w2v_model.vector_size
            num_tokens = 0
            
            for token in tokens:
                if token in self.w2v_model.wv:
                    recipe_vector = [a + b for a, b in zip(recipe_vector, self.w2v_model.wv[token])]
                    num_tokens += 1
            
            if num_tokens > 0:
                recipe_vector = [x / num_tokens for x in recipe_vector]
            
            all_recipes_vector.append(recipe_vector)
        
        self.all_recipes_vector = np.array(all_recipes_vector)
        print("[OK] Recipe vectors computed!")

    def user_input_vectorize(self, user_input):
        """Convert user input to vector and identify unknown words"""
        user_input_tokens = word_tokenize(user_input.lower())
        
        # Remove punctuation from tokens to avoid flagging ',' as unknown
        user_input_tokens = [t for t in user_input_tokens if t not in string.punctuation]

        user_input_vector = [0] * self.w2v_model.vector_size
        num_tokens = 0
        unknown_words = []
        
        for token in user_input_tokens:
            if token in self.w2v_model.wv:
                user_input_vector = [a + b for a, b in zip(user_input_vector, self.w2v_model.wv[token])]
                num_tokens += 1
            else:
                unknown_words.append(token)
        
        if num_tokens > 0:
            user_input_vector = [x / num_tokens for x in user_input_vector]
        
        return user_input_vector, unknown_words

    def recommend(self, user_input, top_n=5):
        """Get recipe recommendations based on user input"""
        user_input_vector, unknown_words = self.user_input_vectorize(user_input)
        user_input_vector = np.array(user_input_vector).reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(user_input_vector, self.all_recipes_vector)
        
        # Get top N recommendations
        top_indices = similarities.argsort()[0][-top_n:][::-1]
        
        # Extract similarity scores for top N recipes
        top_scores = similarities[0][top_indices].tolist()
        
        # Calculate average cosine similarity
        avg_similarity = float(np.mean(top_scores))
        
        return top_indices, unknown_words, top_scores, avg_similarity

    def display_recommendations(self, indices):
        """Display recipe recommendations"""
        print("\n" + "="*80)
        print("TOP RECIPE RECOMMENDATIONS")
        print("="*80)
        
        for i, idx in enumerate(indices, 1):
            print(f"\n{i}. {self.df_clean['title'][idx]}")
            print(f"   {'─'*76}")
            print(f"   Ingredients: {self.df_clean['items'][idx][:200]}...")
            print(f"   Instructions: {self.df_clean['instructions'][idx][:200]}...")
        
        print("\n" + "="*80)

    def save_model(self, filename='recipe_model.pkl'):
        """Save trained model to file"""
        print(f"\n>> Saving model to {filename}...")
        with open(filename, 'wb') as f:
            pickle.dump({
                'df_clean': self.df_clean,
                'w2v_model': self.w2v_model,
                'all_recipes_vector': self.all_recipes_vector,
                'all_tokens': self.all_tokens
            }, f)
        print("[OK] Model saved!")

    def load_model(self, filename='recipe_model.pkl'):
        """Load trained model from file"""
        print(f"\n>> Loading model from {filename}...")
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.df_clean = data['df_clean']
            self.w2v_model = data['w2v_model']
            self.all_recipes_vector = data['all_recipes_vector']
            self.all_tokens = data['all_tokens']
        print("[OK] Model loaded!")


def main():
    """Main function to run the recipe recommender"""
    print("\n" + "="*80)
    print("RECIPE RECOMMENDER SYSTEM")
    print("="*80)
    
    recommender = RecipeRecommender()
    
    # Check if saved model exists
    model_file = 'recipe_model.pkl'
    
    if os.path.exists(model_file):
        print(f"\n[OK] Found saved model: {model_file}")
        choice = input("Load saved model? (yes/no) [yes]: ").lower().strip() or 'yes'
        
        if choice in ['yes', 'y']:
            recommender.load_model(model_file)
        else:
            recommender.load_data()
            recommender.train_model()
            
            save = input("\nSave model for future use? (yes/no) [yes]: ").lower().strip() or 'yes'
            if save in ['yes', 'y']:
                recommender.save_model(model_file)
    else:
        recommender.load_data()
        recommender.train_model()
        
        save = input("\nSave model for future use? (yes/no) [yes]: ").lower().strip() or 'yes'
        if save in ['yes', 'y']:
            recommender.save_model(model_file)
    
    # Interactive loop
    print("\n" + "="*80)
    print("Enter ingredients you have (comma-separated)")
    print("Type 'quit' or 'exit' to stop")
    print("="*80)
    
    while True:
        user_input = input("\nYour ingredients: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for using Recipe Recommender! Happy cooking!")
            break
        
        if not user_input:
            print("[ERROR] Please enter some ingredients!")
            continue
        
        try:
            indices = recommender.recommend(user_input, top_n=5)
            recommender.display_recommendations(indices)
        except Exception as e:
            print(f"[ERROR] {e}")
            print("Please try again with different ingredients.")


if __name__ == "__main__":
    main()
