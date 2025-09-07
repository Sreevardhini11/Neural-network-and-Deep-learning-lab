# 1. Install TensorFlow (do this only in terminal, not in script)
# pip install tensorflow

import numpy as np
import tensorflow as tf

# 2. Define OR and XOR datasets
X_OR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_OR = np.array([[0], [1], [1], [1]], dtype=np.float32)

X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_XOR = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 3. Function to create and train a single-layer perceptron
def train_perceptron(X, y, epochs=100, learning_rate=0.1):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, y, epochs=epochs, verbose=0)
    return model

# 4. Train on OR gate
model_OR = train_perceptron(X_OR, y_OR, epochs=500, learning_rate=0.5)
loss_OR, accuracy_OR = model_OR.evaluate(X_OR, y_OR, verbose=0)
print(f"✅ OR Gate Accuracy: {accuracy_OR:.2f}")

# 5. Train on XOR gate (not possible with single-layer perceptron, but we try)
model_XOR = train_perceptron(X_XOR, y_XOR, epochs=1000, learning_rate=0.8)
loss_XOR, accuracy_XOR = model_XOR.evaluate(X_XOR, y_XOR, verbose=0)
print(f"❌ XOR Gate Accuracy: {accuracy_XOR:.2f} (Single-layer perceptron cannot learn XOR)")

# 6. Make a prediction using model_OR
input1 = 0
input2 = 0
user_input = np.array([[input1, input2]], dtype=np.float32)
prediction = model_OR.predict(user_input, verbose=0)

if prediction.item() > 0.5:
    print(f"The model predicts 1 for input ({input1}, {input2}).")
else:
    print(f"The model predicts 0 for input ({input1}, {input2}).")
