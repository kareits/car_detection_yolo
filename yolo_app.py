import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# Заголовок приложения
st.set_page_config(page_title="Car Detection App", layout="wide")
st.title("🚗 YOLO Car Detection App")

st.write("Загрузите изображение автомобиля для проверки")

# Кэшируем модель, чтобы не загружать каждый раз
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # можно заменить на yolov8s.pt или свою модель

model = load_model()

# Загрузка изображения
uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("📷 Загруженное изображение")
    st.image(image, use_column_width=True)

    # Конвертируем в numpy
    image_np = np.array(image)

    # Предсказание
    with st.spinner("🔍 Анализ изображения..."):
        results = model(image_np)

    result = results[0]

    # Рисуем bounding boxes
    annotated_frame = result.plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    st.subheader("🎯 Результат детекции")
    st.image(annotated_frame, use_column_width=True)

    # Формируем таблицу предсказаний
    boxes = result.boxes
    data = []

    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        data.append({
            "Class": class_name,
            "Confidence": round(confidence, 3),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
        })

    if data:
        df = pd.DataFrame(data)
        st.subheader("📊 Детали предсказания")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Объекты не обнаружены.")
