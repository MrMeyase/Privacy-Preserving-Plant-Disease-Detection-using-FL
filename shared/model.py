import tensorflow as tf
from tensorflow.keras import layers, models

from shared.config import IMG_SIZE, NUM_CLASSES, DROPOUT_RATE, LEARNING_RATE

def setup_tensorflow_for_gpu(enable_mixed_precision=True):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            if enable_mixed_precision:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float16")

            print(f"GPU detected: {len(gpus)} device(s)")
            print("Mixed precision:", "ON" if enable_mixed_precision else "OFF")

        except RuntimeError as e:
            print("GPU config error:", e)
    else:
        print("No GPU detected. Running on CPU.")

def build_model(
    num_classes=NUM_CLASSES,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    dropout_rate=DROPOUT_RATE,
    train_base=False,
):
    """
    MobileNetV2-based classifier for plant disease detection.

    Notes:
    - Input images are assumed to be raw RGB images in [0, 255].
    - Rescaling is done inside the model so the Pi inference script can
      send images without manual normalization.
    """

    inputs = layers.Input(shape=input_shape, name="input_image")

    # Keep this so your Pi inference script can skip manual normalization
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base_model.trainable = train_base

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="plant_disease_mobilenetv2")
    return model


def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compile the model for multi-class classification.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_base_model(model, fine_tune_at=None):
    """
    Unfreeze the base model for fine-tuning.

    If fine_tune_at is provided, layers before that index stay frozen.
    """
    base_model = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2"):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Base model not found inside the model.")

    base_model.trainable = True

    if fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    return model