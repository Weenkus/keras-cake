from keras.datasets import imdb
from keras.layers import GRU, Activation, Dense, Flatten, Embedding, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing import sequence

from capsule import Capsule

max_features = 20000
maxlen = (
    80
)  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print("Build model...")
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(SpatialDropout1D(0.2))
model.add(GRU(128, dropout=0.2, return_sequences=True))
model.add(Capsule(num_capsule=10, dim_capsule=16, routings=5, share_weights=True))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation("sigmoid"))

# try using different optimizers and different optimizer configs
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
print(len(X_train), "train sequences")
print(len(X_test), "test sequences")

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("Train...")
model.fit(
    X_train, y_train, batch_size=batch_size, epochs=15, validation_data=(X_test, y_test)
)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Test score:", score)
print("Test accuracy:", acc)
