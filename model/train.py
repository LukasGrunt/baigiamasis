from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.cnn_model import build_cnn

def train_model(X_train, y_train, X_val, y_val, class_weights, batch_size=64, learning_rate=0.0002, epochs=18):
    model = build_cnn()
    model.compile(optimizer=AdamW(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    datagen = ImageDataGenerator(
        rotation_range=35,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow(X_train, y_train, batch_size=batch_size)

    history = model.fit(
        train_gen,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[early_stopping],
        class_weight=class_weights
    )

    return model, history
