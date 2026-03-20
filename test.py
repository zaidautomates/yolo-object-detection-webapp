import requests

# The endpoint of your local Flask server
url = 'http://127.0.0.1:5000/predict'
image_path = 'test_image.jpg' 

print(f"Sending '{image_path}' to the YOLO API...")

try:
    with open(image_path, 'rb') as img_data:
        response = requests.post(url, files={'image': img_data})
    print("\nAPI Response received:")
    print(response.json())
except FileNotFoundError:
    print(f"ERROR: Cannot find {image_path}. Please place an image named test_image.jpg in the same directory.")
except requests.exceptions.ConnectionError:
    print("ERROR: Connection Refused. Did you start `python api.py` in another terminal?")
