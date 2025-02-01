import keras.src.saving
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os

camera_nr = 0

# Mapa etykiet
labels = ['fist', 'five', 'none', 'okay', 'peace', 'rad', 'straight', 'thumbs']

# Funkcja do wczytywania modelu
def load_model():
    model = keras.src.saving.load_model("gest_recogn_data_final.h5")
    return model


def get_frame_size():
    frame_size = 128
    return frame_size

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
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    # Znajdowanie konturów
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)

    if contours:
        # Największy kontur
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

    # Dodanie "batch" - paczki, do tablicy
    img_array = np.expand_dims(img_array, axis=0)

    # Konwersja na tensor TensorFlow
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Predykcja modelu
    prediction = model(img_tensor, training=False)
    predicted_class = tf.argmax(prediction, axis=1)[0]

    # Pobranie etykiety klasy
    predicted_label = labels[predicted_class.numpy()]

    return predicted_label, prediction[0][predicted_class].numpy(), processed_image

# Funkcja do uruchomienia kamery
def capture_from_camera(frame_size, model):
    # Otwieranie kamery
    cap = cv2.VideoCapture(camera_nr)
    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        return

    while True:
        # Pobranie klatki
        ret, frame = cap.read()
        if not ret:
            break

        # Zmieniamy format obrazu na RGB (dla PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predykcja
        predicted_label, confidence, model_input = predict_image(frame_rgb, model, frame_size)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Zapisz obraz z bieżącego gestu
            image_name = f"{predicted_label}_{int(confidence*100)}.png"
            image_path = os.path.join("saved_images", image_name)
            if not os.path.exists("saved_images"):
                os.makedirs("saved_images")
            cv2.imwrite(image_path, frame)
            print(f"Zapisano obraz: {image_path}")

        # Wyświetlenie wyniku

        cv2.putText(frame, f"{predicted_label} ({confidence*100:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Kamera', frame)
        cv2.imshow('Model Input', model_input)

        # Sprawdzenie, czy naciśnięto spację


        # Zakończenie działania programu po naciśnięciu 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = load_model()
    frame_size = get_frame_size()
    capture_from_camera(frame_size, model)
