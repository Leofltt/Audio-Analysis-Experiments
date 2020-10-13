import tensorflow.keras as k
from preprocess import gen_train_seqs, SEQ_LENGTH

OUTPUT_SZ = 38  # vocab size
NUM_UNITS = [256]
LOSS = 'sparse_categorical_crossentropy'
LR = 0.001
EPOCHS = 50
BATCH_SZ = 64
SAVE_PATH = 'model.h5'

def buil_model(num_outputs, num_units, loss, learning_rate):
    input = k.layers.Input(shape=(None, num_outputs))
    x = k.layers.LSTM(num_units[0])(input)
    x = k.layers.Dropout(0.25)(x)
    output = k.layers.Dense(num_outputs, activation='softmax')(x)
    model = k.Model(input, output)
    model.compile(loss=loss, optimizer=k.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    model.summary()
    return model


def train(num_outputs=OUTPUT_SZ, num_units=NUM_UNITS, loss=LOSS, learning_rate=LR, save_path=SAVE_PATH):
    inputs, targets = gen_train_seqs(SEQ_LENGTH)
    model = buil_model(num_outputs, num_units, loss, learning_rate)
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SZ)
    model.save(save_path)

