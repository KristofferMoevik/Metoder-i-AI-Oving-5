import pickle
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import numpy as np


with open(file="keras-data.pickle", mode="rb") as file:
    data = pickle.load(file)

padded_x_train = tf.keras.preprocessing.sequence.pad_sequences(
    data["x_train"], maxlen=10
)

model = tf.keras.Sequential()
input_dim = data["max_length"]
output_dim = 10
# model.add(tf.keras.layers.Embedding(input_dim,
#     output_dim,
#     embeddings_initializer='uniform',
#     embeddings_regularizer=None,
#     activity_regularizer=None,
#     embeddings_constraint=None,
#     mask_zero=False,
#     input_length= len(padded_x_train))
# )
# lstm = tf.keras.layers.LSTM(4)
# model.add(tf.keras.layers.LSTM(1, input_shape=(10,1)))
# model.add(tf.keras.layers.Dense(32, activation="sigmoid"))
# model.compile(optimizer='adam', loss='mse')
model.add(tf.keras.layers.Embedding(data["max_length"], output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(tf.keras.layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(tf.keras.layers.Dense(10))
model.compile(optimizer="adam", loss="mse")
model.save("save_unfitted")
model.fit(np.array(padded_x_train), np.array(data["y_train"]), epochs=1)
model.save("save_fitted")
print(model.evaluate(data["x_test"], data["y_test"]))

