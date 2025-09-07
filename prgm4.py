import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. Load and preprocess IMDB dataset
# =========================
max_features = 10000  # Vocabulary size
maxlen = 200          # Sequence length

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# =========================
# 2. Define LSTM model
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_features, output_dim=128, input_length=maxlen),
    tf.keras.layers.LSTM(128, return_sequences=False),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# =========================
# 3. Train the model
# =========================
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# =========================
# 4. Plot training & validation accuracy
# =========================
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('LSTM Training vs Validation Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('LSTM Training vs Validation Loss')
plt.legend()
plt.show()

# =========================
# 5. Evaluate model performance
# =========================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# =========================
# 6. User input prediction
# =========================
def preprocess_input(text, tokenizer, maxlen):
    """Tokenize and pad a single user input text"""
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    return padded_sequence

# Create a tokenizer based on IMDB dataset (word index)
word_index = imdb.get_word_index()
tokenizer = Tokenizer(num_words=max_features)
tokenizer.word_index = word_index

user_input = input("Enter a movie review: ")
processed_input = preprocess_input(user_input, tokenizer, maxlen)

prediction = model.predict(processed_input)
sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
print(f"Predicted Sentiment: {sentiment} (Probability: {prediction[0][0]:.2f})")
