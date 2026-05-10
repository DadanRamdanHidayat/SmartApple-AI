import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32

# =========================
# LOAD DATASET (FOLDER)
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int'  # cocok untuk sparse_categorical_crossentropy
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# Ambil nama kelas (PENTING buat prediksi nanti)
class_names = train_ds.class_names
print("Kelas:", class_names)

# =========================
# NORMALISASI
# =========================
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
valid_ds = valid_ds.map(lambda x, y: (normalization_layer(x), y))

# =========================
# PREFETCH (BIAR CEPAT)
# =========================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)

# =========================
# MODEL (MobileNetV2)
# =========================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)

# 🔥 UBAH JADI 4 KELAS
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

# =========================
# COMPILE
# =========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAINING
# =========================
model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10
)

# =========================
# SAVE MODEL
# =========================
model.save("model/model.h5")

print("✅ Model berhasil disimpan!")