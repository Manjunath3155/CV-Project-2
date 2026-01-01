import cv2
import numpy as np
import base64
from pyzbar.pyzbar import decode
import requests
import io
from PIL import Image

class ScannerService:
    def scan_barcode(self, image_data_b64):
        """
        Scans an image for a barcode and returns the product name.
        """
        try:
            # Decode image
            if "base64," in image_data_b64:
                image_data_b64 = image_data_b64.split("base64,")[1]
            
            image_bytes = base64.b64decode(image_data_b64)
            
            # Convert to numpy array for cv2
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) # Grayscale for better detection
            
            if img is None:
                print("Failed to decode image for barcode scanning")
                return None

            # Detect barcodes
            barcodes = decode(img)
            
            if not barcodes:
                return None
            
            # Use the first barcode found
            barcode_data = barcodes[0].data.decode('utf-8')
            print(f"Barcode detected: {barcode_data}")
            
            # Fetch details
            return self.fetch_product_name(barcode_data)
            
        except Exception as e:
            print(f"Barcode Scan Error: {e}")
            return None

    def fetch_product_name(self, barcode):
        """
        Fetches product name from OpenFoodFacts
        """
        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                product = data.get('product', {})
                # Try generic name first, then product name
                name = product.get('generic_name_en') or product.get('product_name') or product.get('product_name_en')
                return name
            return None
        except Exception as e:
            print(f"OpenFoodFacts Error: {e}")
            return None

