import tensorflow as tf
from tensorflow.keras import layers, models


def build_model():
    model = models.Sequential([
        # Normalization
        layers.Rescaling(1./255, input_shape=(150, 150, 3)),

        # Augmentation (applied during training only)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.2),

        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),

        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),

        # Conv Block 4
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),      # mirrors PyTorch AdaptiveAvgPool

        # Classifier
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(6, activation='softmax')
    ])

    return model


def train():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "data/seg_train",
        image_size=(150, 150),              
        batch_size=32,
        label_mode='int',
        shuffle=True
    )

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    print(f"Training on: {device}")

    model = build_model()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    model.fit(train_ds, epochs=10)

    model.save("angelah_model.keras")
    print("Model saved to angelah_model.keras")


if __name__ == "__main__":
    train()
