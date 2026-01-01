# 🍳 Kenshoku AI (食)
### NLP-Powered Intelligent Recipe Recommender & Pantry Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Kenshoku AI** is a modern, full-stack application designed to solve the "What should I cook today?" dilemma. By combining Natural Language Processing (NLP) with Computer Vision, it transforms your available ingredients into delicious recipe suggestions.

---

## ✨ Key Features

- 🧠 **NLP Recommender Engine**: Utilizes **Word2Vec** embeddings and **Cosine Similarity** to match your ingredients against a database of over **125,000 recipes**.
- 📸 **AI Pantry Scanner**: Snap a photo of your fridge or pantry, and the system automatically identifies ingredients using **Llama-3 Vision** (via Groq).
- 🛒 **Barcode Integration**: Instantly identify packaged goods by scanning barcodes, fetching data directly from the **OpenFoodFacts API**.
- 📺 **Visual Learning**: Automatically fetches high-quality **YouTube recipe tutorials** for every recommendation.
- 🎨 **Modern UI/UX**: A sleek, responsive dashboard built with **React** and **GSAP** for fluid animations.

---

## 🏗️ Technical Architecture

### Backend (Python/FastAPI)
- **Data Processing**: Scikit-learn and Pandas handle the cleanup of a massive 125k recipe dataset.
- **NLP Model**: Gensim-based **Word2Vec** creates high-dimensional vector representations of recipe content (Title + Ingredients + Instructions).
- **Vision Services**: Groq-powered Vision API for ingredient detection and PyZbar for local barcode processing.
- **Search Logic**: Multi-stage filtering and similarity ranking to ensure highly relevant results.

### Frontend (React/Vite)
- **State Management**: Modern React hooks and Context API.
- **Animations**: **GSAP** (GreenSock) for high-performance transitions and UI polish.
- **Styling**: **Sass (SCSS)** with a modular architecture for a clean, consistent design.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 18+
- [Groq API Key](https://console.groq.com/) (for Vision features)
- [YouTube Data API v3 Key](https://developers.google.com/youtube/v3) (for recipe videos)

### 1. Backend Setup
```bash
# Navigate to root
pip install -r requirements.txt

# Create .env file
# GROQ_API_KEY=your_key
# YOUTUBE_API_KEY=your_key

# Start the server
python main.py
```

### 2. Frontend Setup
```bash
cd client
npm install
npm run dev
```

---

## 📊 Dataset Information
The project utilizes a comprehensive dataset of 125,000 recipes scraped from:
- AllRecipes
- Epicurious
- Food Network

*Note: Due to size constraints, the raw JSON datasets and the pre-trained `recipe_model.pkl` (400MB+) are ignored in this repository. The system will automatically attempt to train the model on first launch if data is available.*

---

## 🛠️ Future Roadmap
- [ ] User profiles and saved "favorite" recipes.
- [ ] Dietary preference filters (Vegan, Keto, Gluten-free).
- [ ] Shopping list generation from selected recipes.
- [ ] Mobile app version using React Native.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Developed with ❤️ by [Your Name/GitHub Handle]
