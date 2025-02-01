import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array

# Ładowanie danych
data_path = "train_data2\validation"
model_path = "gest_recogn_data_final.h5"
image_data = []
labels = []

# Ładowanie obrazów
for folder_name in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder_name)
    if os.path.isdir(folder_path):  # Sprawdzenie, czy to folder
        for image_name in os.listdir(folder_path):
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(folder_path, image_name)
                try:
                    image = load_img(image_path, color_mode='grayscale', target_size=(128, 128))
                    image = img_to_array(image) / 255.0
                    image_data.append(image)
                    labels.append(folder_name)  # Użycie nazw folderów jako etykiet
                except Exception as e:
                    print(f"Błąd ładowania obrazu {image_path}: {e}")
                    continue # pomijamy uszkodzony obraz i idziemy dalej

image_data = np.array(image_data)

# Konwersja etykiet na wartości numeryczne
unique_labels = sorted(list(set(labels)))
label_mapping = {label: i for i, label in enumerate(unique_labels)}
reverse_mapping = {i: label for label, i in label_mapping.items()}
numerical_labels = np.array([label_mapping[label] for label in labels])

# Konwersja etykiet na format one-hot encoding
numerical_labels_categorical = tf.keras.utils.to_categorical(numerical_labels)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    image_data, numerical_labels_categorical, test_size=0.2, random_state=44, stratify=numerical_labels
)

# Załaduj wytrenowany model (jeśli już jest zapisany)
model = tf.keras.models.load_model(model_path)

# Predykcje na zbiorze testowym
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Prawdziwe etykiety
y_test_classes = np.argmax(y_test, axis=1)

# Obliczanie dokładności
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f'Dokładność: {accuracy:.4f}')

# Obliczanie F1-score
f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
print(f'F1-score: {f1:.4f}')

# Obliczanie Recall
recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
print(f'Recall: {recall:.4f}')

# Generowanie pełnego raportu klasyfikacji
report = classification_report(y_test_classes, y_pred_classes, target_names=unique_labels)
print("Raport klasyfikacji:\n", report)

# Macierz pomyłek
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Wykres matrycy pomyłek
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Przewidywane etykiety')
plt.ylabel('Prawdziwe etykiety')
plt.title('Macierz pomyłek')
plt.show()

