# 🚗 YOLO Car Detection App

This is a Streamlit web application for detecting cars in images using YOLOv8. The application allows users to upload an image and receive object detection results displayed directly in the browser.

The app performs the following steps: a user uploads an image (jpg, jpeg, png, or jfif), the backend sends the image to a YOLOv8 model for inference, and the detection results (bounding boxes, class labels, and confidence scores) are returned and visualized on the frontend.

## Technologies Used

- Python 3.9+
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- Pillow
- NumPy
- Pandas

## Project Structure

car_detector_app/
│
├── app.py
├── requirements.txt
└── README.md

## Installation

First, clone the repository:

git clone <your-repository-url>
cd car_detector_app

Create and activate a virtual environment (recommended):

On Windows:
python -m venv venv
venv\Scripts\activate

On macOS/Linux:
python -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

If requirements.txt is not available, install manually:

pip install streamlit ultralytics pillow opencv-python numpy pandas

## Running the Application

Start the Streamlit server with:

streamlit run app.py

After running the command, open your browser and go to:

http://localhost:8501

## Supported Image Formats

The application supports the following image formats:
- jpg
- jpeg
- png
- jfif

JFIF files are automatically processed as standard JPEG images.

## How It Works

1. The user uploads an image through the Streamlit interface.
2. The image is converted into a NumPy array.
3. The YOLOv8 model performs inference.
4. The model returns detected objects with:
   - Class name
   - Confidence score
   - Bounding box coordinates
5. The annotated image with bounding boxes is displayed.
6. A table with prediction details is shown below the image.

## Default Model

By default, the application uses:

yolov8n.pt

This is the nano version of YOLOv8, optimized for speed. You can replace it with other versions such as:

- yolov8s.pt (more accurate)
- yolov8m.pt
- yolov8l.pt
- A custom-trained model

To change the model, modify this line in app.py:

model = YOLO("yolov8n.pt")

## Example Output

The results table contains:

- Class (e.g., car)
- Confidence (e.g., 0.93)
- Bounding box coordinates (x1, y1, x2, y2)

## Possible Improvements

- Validate that exactly one car is detected
- Enforce minimum object size in the image
- Add license plate OCR
- Add damage detection (custom training)
- Separate YOLO into a FastAPI microservice
- Add Docker support
- Implement logging and fraud detection checks

## Generating requirements.txt

To export project dependencies:

pip freeze > requirements.txt

## License

This project is intended for educational and demonstration purposes.

## Author

@kareits
