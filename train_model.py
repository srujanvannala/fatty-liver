import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ----------------------------
# Function to load trained model
# ----------------------------
def load_trained_model(model_path="liver_model.h5"):
    """
    Load a trained liver classification model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    model = tf.keras.models.load_model(model_path)
    return model


# ----------------------------
# Function to train the model
# ----------------------------
def train_model(
    data_dir="data",
    img_size=(224, 224),
    batch_size=16,
    epochs=10
):
    """
    Train a CNN model for liver ultrasound classification
    using images inside 'data/' folder.
    """

    # Automatically count number of classes
    num_classes = len([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    print(f"Detected {num_classes} classes:", [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

    # ------------------------
    # Image data generators
    # ------------------------
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    # ------------------------
    # CNN Model
    # ------------------------
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')  # dynamic output
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # ------------------------
    # Train the model
    # ------------------------
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )

    # Save the model
    model.save("liver_model.h5")
    print("Model saved as liver_model.h5")

    return model, history


# ----------------------------
# Run training if executed directly
# ----------------------------
if __name__ == "__main__":
    train_model(data_dir="data", img_size=(224,224), batch_size=16, epochs=10)
