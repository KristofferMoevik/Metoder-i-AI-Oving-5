import pickle
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import numpy as np

model = tf.keras.models.load_model("save_fitted")
with open(file="keras-data.pickle", mode="rb") as file:
    data = pickle.load(file)

padded_x_test = tf.keras.preprocessing.sequence.pad_sequences(data["x_test"], maxlen=10)

print(model.evaluate(padded_x_test, np.array(data["y_test"])))
# with input max 100 got 0.1215