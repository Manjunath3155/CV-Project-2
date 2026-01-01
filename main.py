from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
from dotenv import load_dotenv
from recipe_recommender import RecipeRecommender
from services.vision import VisionService
from services.scanner import ScannerService

load_dotenv()

app = FastAPI(title="KENSHOKU AI API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
recommender = RecipeRecommender()
vision_service = VisionService()
scanner_service = ScannerService()

@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts"""
    print("Initializing Recipe Recommender...")
    model_file = 'recipe_model.pkl'
    
    if os.path.exists(model_file):
        print(f"Loading saved model from {model_file}...")
        recommender.load_model(model_file)
    else:
        print("No saved model found.")
        if os.path.exists("recipes_raw_nosource_ar.json"):
            print("Training new model (this may take a few minutes)...")
            recommender.load_data()
            recommender.train_model()
            recommender.save_model(model_file)
        else:
            print("ERROR: Data files not found.")

class RecipeRequest(BaseModel):
    ingredients: str

class ImageRequest(BaseModel):
    image: str # Base64 encoded image

class RecipeResponse(BaseModel):
    id: int
    title: str
    ingredients: str
    instructions: str
    score: Optional[float] = None

class RecommendationResponse(BaseModel):
    results: List[RecipeResponse]
    analysis: dict

@app.get("/")
def read_root():
    return {"status": "online", "message": "KENSHOKU AI API is running"}

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecipeRequest):
    if not recommender.w2v_model:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        # recommend now returns tuple (indices, unknown_words, scores, avg_similarity)
        indices, unknown_words, scores, avg_similarity = recommender.recommend(request.ingredients, top_n=10)
        
        results = []
        for idx, score in zip(indices, scores):
            try:
                recipe = {
                    "id": int(idx),
                    "title": recommender.df_clean['title'][idx],
                    "ingredients": str(recommender.df_clean['items'][idx]), 
                    "instructions": str(recommender.df_clean['instructions'][idx]),
                    "score": round(float(score), 4)  # Round to 4 decimal places
                }
                results.append(recipe)
            except Exception as e:
                continue
                
        return {
            "results": results,
            "analysis": {
                "unknown_ingredients": unknown_words,
                "all_unknown": len(unknown_words) > 0 and len(results) == 0,
                "average_similarity": round(avg_similarity, 4)  # Average cosine similarity score
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recipe/{recipe_id}/video")
def get_recipe_video(recipe_id: int):
    try:
        if recommender.df_clean is None:
             raise HTTPException(status_code=503, detail="Model/Data not loaded.")
             
        recipe_title = recommender.df_clean['title'][recipe_id]
        
        # Search YouTube
        youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        if not youtube_api_key:
            print("[WARN] YOUTUBE_API_KEY not found.")
            return {"videoId": None}

        search_query = f"{recipe_title} recipe"
        print(f"[DEBUG] Searching YouTube for: {search_query}")
        
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": search_query,
                "type": "video",
                "maxResults": 1,
                "key": youtube_api_key
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('items'):
                video_id = data['items'][0]['id']['videoId']
                return {"videoId": video_id, "url": f"https://www.youtube.com/watch?v={video_id}"}
        else:
            print(f"[ERROR] YouTube API Error: {response.text}")
            
        return {"videoId": None}
    except Exception as e:
        print(f"[ERROR] Video fetch error: {e}")
        return {"videoId": None}

@app.post("/scan/pantry")
def scan_pantry(request: ImageRequest):
    """Uses Gemini Vision to identify ingredients in an image"""
    if not vision_service.client:
        raise HTTPException(status_code=503, detail="Gemini API Key missing or client failed.")
    
    ingredients = vision_service.detect_ingredients(request.image)
    if not ingredients:
        return {"ingredients": [], "message": "No ingredients detected."}
    
    return {"ingredients": ingredients, "message": "Success"}

@app.post("/scan/barcode")
def scan_barcode(request: ImageRequest):
    """Scans for a barcode and returns product name"""
    product_name = scanner_service.scan_barcode(request.image)
    
    if not product_name:
         return {"product": None, "message": "No barcode detected or product not found."}
    
    return {"product": product_name, "message": "Success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
