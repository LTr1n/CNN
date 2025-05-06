import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# =================== 1️⃣ Data Augmentation ===================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

# =================== 2️⃣ Load dữ liệu ===================
train_dir = "/kaggle/input/1234567/train"
val_dir = "/kaggle/input/1234567/train"

batch_size = 32
img_size = (224, 224)

autotune = tf.data.AUTOTUNE

train_dataset = keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=batch_size,
    image_size=img_size
).map(lambda x, y: (data_augmentation(x) / 255.0, y), num_parallel_calls=autotune).prefetch(autotune)

val_dataset = keras.utils.image_dataset_from_directory(
    val_dir,
    shuffle=False,
    batch_size=batch_size,
    image_size=img_size
).map(lambda x, y: (x / 255.0, y), num_parallel_calls=autotune).prefetch(autotune)

# =================== 3️⃣ Mô hình CNN nâng cao ===================
def create_model(num_classes=10):
    model = keras.Sequential([
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),  # Giảm overfitting
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

model = create_model(num_classes=10)

# =================== 4️⃣ Cấu hình tối ưu hóa ===================
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-5),
    metrics=["accuracy"]
)

# =================== 5️⃣ Training với callback ===================
epochs = 40

callbacks = [
keras.callbacks.ModelCheckpoint("/kaggle/working/best_model.keras", save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks
)

# =================== 6️⃣ Hiển thị biểu đồ ===================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training & Validation Accuracy')

plt.show()