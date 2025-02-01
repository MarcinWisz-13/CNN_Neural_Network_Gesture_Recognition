import os
import cv2
import numpy as np
import tensorflow as tf
import keras.src.saving

# Lista etykiet
labels = ['fist', 'five', 'none', 'okay', 'peace', 'rad', 'straight', 'thumbs']

folder_path = "saved_images"
model_path = "gest_recogn_data_final.h5"


def load_model():
    model = keras.src.saving.load_model(model_path)
    return model


def get_frame_size():
    frame_size = 128
    return frame_size

# Funkcja do wykrywania dłoni
def preprocess_hand(image, frame_size):
    # Konwersja do skali szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Redukcja szumów
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Progowanie Otsu dla lepszej segmentacji dłoni
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Odwrócenie progowania, aby dłoń była biała
    thresh = cv2.bitwise_not(thresh)

    # Operacje morfologiczne, aby wzmocnić kontury
    kernel = np.ones((5, 5), np.uint8)  # Mniejsze jądro dla delikatniejszego rozszerzenia
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)  # Mniejsza liczba iteracji
    dilated = cv2.dilate(closed, kernel, iterations=1)  # Mniejsza liczba iteracji

    # Znajdowanie konturów
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)

    if contours:
        # Największy kontur (zakładamy, że to dłoń)
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 3000:  # Mniejsza wrażliwość na drobne kontury
            # Wypełnienie konturu
            cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)

            # Obrysowanie konturu cieńszą linią (grubość zmniejszona)
            cv2.drawContours(mask, [max_contour], -1, 255, thickness=2)  # Grubość zmniejszona do 2

    # Skalowanie do wymiaru docelowego
    result_resized = cv2.resize(mask, (frame_size, frame_size))

    return result_resized

# Funkcja do predykcji z obrazu
def predict_image(image, model, frame_size):
    # Preprocess dłoni (wybielenie dłoni i czarne tło)
    processed_image = preprocess_hand(image, frame_size)

    # Normalizacja obrazu do zakresu [0, 1]
    img_array = processed_image / 255.0

    # Dodajemy wymiar "batch"
    img_array = np.expand_dims(img_array, axis=0)

    # Konwersja na tensor TensorFlow
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Predykcja modelu
    prediction = model(img_tensor, training=False)
    predicted_class = tf.argmax(prediction, axis=1)[0]

    # Pobranie etykiety klasy
    predicted_label = labels[predicted_class.numpy()]

    return predicted_label, prediction[0][predicted_class].numpy(), processed_image

# Funkcja do przetwarzania obrazów w folderze
def process_images_in_folder(folder_path, model, frame_size):
    # Pobranie wszystkich plików w folderze
    for filename in os.listdir(folder_path):
        # Sprawdzenie, czy plik jest obrazem (np. JPG, PNG)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Predykcja
            predicted_label, confidence, model_input = predict_image(image, model, frame_size)

            # Wypisanie wyniku do konsoli
            print(f"Obraz: {filename} | Predykcja: {predicted_label} ({confidence*100:.2f}%)")

if __name__ == "__main__":
    model = load_model()
    frame_size = get_frame_size()

    process_images_in_folder(folder_path, model, frame_size)
