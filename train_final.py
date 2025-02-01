import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Ścieżka do danych i nazwa modelu
data_path = r"train"
model_name_to_save = "gest_recogn_data_final.h5"

# Listy na dane i etykiety
image_data = []
labels = []

# Ładowanie danych
for folder_name in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder_name)
    if os.path.isdir(folder_path):  # Sprawdzenie, czy to folder
        for image_name in os.listdir(folder_path):
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(folder_path, image_name)
                try: # Dodajemy obsługę wyjątków, gdyby obraz był uszkodzony.
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

numerical_labels_categorical = tf.keras.utils.to_categorical(numerical_labels)

# Podział na zbiór treningowy i testowy - PO WŁADWANIU DANYCH
X_train, X_test, y_train, y_test = train_test_split(
    image_data, numerical_labels_categorical, test_size=0.2, random_state=44, stratify=numerical_labels
)

# Augmentacja danych
datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)  # Dopasowanie do danych treningowych

# Model -
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(unique_labels), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    epochs=35,
                    validation_data=(X_test, y_test))

model.save(model_name_to_save)

# Wykresy i Ewaluacja

# Testowanie danych
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {test_accuracy*100:.2f}%')

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(classification_report(y_true, y_pred_classes, target_names=unique_labels)) # Użycie nazw etykiet

misclassified_indices = np.where(y_pred_classes != y_true)[0]
plt.figure(figsize=(20, 4))
for i, idx in enumerate(misclassified_indices[:5]):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[idx].reshape(128, 128, 1), cmap='gray') # Dodano cmap='gray', jeśli obrazy są w skali szarości
    plt.title(f"True: {reverse_mapping[y_true[idx]]}\nPred: {reverse_mapping[y_pred_classes[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
