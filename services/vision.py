import os
import base64
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class VisionService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        print(f"[DEBUG] GROQ_API_KEY loaded: {'Yes' if self.api_key else 'No'}")
        
        if not self.api_key:
            print("[ERROR] GROQ_API_KEY not found in environment variables!")
            print("[ERROR] Make sure .env file exists with GROQ_API_KEY=your_key")
            self.client = None
            return
        
        try:
            print("[DEBUG] Attempting to initialize Groq client...")
            self.client = Groq(api_key=self.api_key)
            self.model = "meta-llama/llama-4-maverick-17b-128e-instruct" 
            print("[SUCCESS] Groq client initialized successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Groq client: {type(e).__name__}: {e}")
            self.client = None

    def detect_ingredients(self, image_data_b64):
        """
        Analyzes an image and returns a list of visible food ingredients.
        """
        if not self.client:
            print("[ERROR] detect_ingredients called but client is None!")
            return {"error": "Groq Client not initialized"}

        try:
            print("[DEBUG] Processing image for ingredient detection (Groq)...")
            
            # Ensure we have a clean base64 string
            if "base64," in image_data_b64:
                # Keep the original for data URL, but strict clean for other uses if needed
                # But for Groq we need the data URL format: data:image/jpeg;base64,...
                # If it's already in that format, great.
                pass
            else:
                # Add header if missing (assuming jpeg)
                image_data_b64 = f"data:image/jpeg;base64,{image_data_b64}"

            print(f"[DEBUG] Image data length: {len(image_data_b64)}")

            prompt = """
            Analyze this image of a fridge, pantry, or food items.
            Identify all distinct food ingredients visible.
            Return ONLY a raw JSON array of strings. 
            Example: ["milk", "eggs", "carrot", "chicken breast"]
            Do not include any markdown formatting like ```json or explanations.
            Just the array.
            """

            print("[DEBUG] Sending request to Groq API...")
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_b64
                                }
                            }
                        ]
                    }
                ],
                model=self.model,
                temperature=0.1,
            )

            text_response = chat_completion.choices[0].message.content.strip()
            print(f"[DEBUG] Groq response: {text_response[:100]}...")
            
            # Clean potential markdown
            if text_response.startswith("```json"):
                text_response = text_response[7:]
            if text_response.endswith("```"):
                text_response = text_response[:-3]
            
            # Additional cleanup for Groq's potential chatter
            start_bracket = text_response.find('[')
            end_bracket = text_response.rfind(']')
            if start_bracket != -1 and end_bracket != -1:
                text_response = text_response[start_bracket:end_bracket+1]

            ingredients = json.loads(text_response)
            print(f"[SUCCESS] Detected {len(ingredients)} ingredients: {ingredients}")
            return ingredients

        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON Parse Error: {e}")
            print(f"[ERROR] Raw response was: {text_response}")
            return []
        except Exception as e:
            print(f"[ERROR] Groq Vision Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return []
