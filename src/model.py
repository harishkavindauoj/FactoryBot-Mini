"""Builds the student model (GRU or CNN)."""
from keras import Model
from tensorflow.keras.layers import Input, GRU, Conv1D, GlobalMaxPooling1D, Dense, Dropout


def build_model(config):
    input_shape = tuple(config['model']['input_shape'])
    model_type = config['model']['type']
    hidden_units = config['model']['hidden_units']
    dropout_rate = config['model']['dropout']

    inp = Input(shape=input_shape)

    if model_type == "GRU":
        x = GRU(hidden_units, return_sequences=False)(inp)
    elif model_type == "CNN":
        x = Conv1D(filters=hidden_units, kernel_size=3, activation='relu')(inp)
        x = GlobalMaxPooling1D()(x)
    else:
        raise ValueError("Unknown model type")

    x = Dropout(dropout_rate)(x)

    risk_output = Dense(4, activation="softmax", name="risk_level")(x)
    ttf_output = Dense(1, activation="linear", name="ttf")(x)

    model = Model(inputs=inp, outputs=[risk_output, ttf_output])
    model.compile(
        loss={"risk_level": "categorical_crossentropy", "ttf": "mse"},
        optimizer="adam",
        metrics={"risk_level": "accuracy", "ttf": "mae"},
    )
    return model