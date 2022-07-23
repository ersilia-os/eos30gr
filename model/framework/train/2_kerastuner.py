import pandas as pd
import numpy as np
import joblib
import shutil
import os
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt

TUNER_PROJECT_NAME_3L = "tuner"
TUNER_MODEL_FOLDER_3L = "tuner"


def process_file(file_name):
    column = "activity80"
    df = pd.read_csv(file_name)
    df[df[column].isnull()] = 0
    cols = [c for c in list(df.columns) if c.startswith("mol2vec-")]
    X = np.array(df[cols])
    y = np.array(df[column])
    return X, y.reshape(-1,1)

Xtr, ytr = process_file("output_file/trainingset_mol2vec.csv")
Xvl, yvl = process_file("output_file/validationset_mol2vec.csv")
Xte, yte = process_file("output_file/testset_mol2vec.csv")

X = np.vstack([Xtr, Xvl, Xte])
y = np.vstack([ytr, yvl, yte])


class TunerClassifier(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.input_shape = X.shape[1]
        self.output_shape = y.shape[1]

    def _model_builder(self, hp):
        model = keras.Sequential()

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
        hp_units2 = hp.Int("units", min_value=32, max_value=512, step=32)
        hp_units3 = hp.Int("units", min_value=32, max_value=512, step=32)
        model.add(
            keras.layers.Dense(
                units=hp_units, activation="relu", input_shape=(self.input_shape,)
            )
        )
        model.add(keras.layers.Dropout(0.5))
        model.add(
            keras.layers.Dense(
                units=hp_units2, activation="relu")
        )
        model.add(keras.layers.Dropout(0.5))
        model.add(
            keras.layers.Dense(
                units=hp_units3, activation="relu")
        )
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(self.output_shape, activation="sigmoid"))

        hp_learning_rate = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss="binary_crossentropy",
            metrics=None,
        )

        return model

    def _search(self, X, y):
        self.tuner = kt.Hyperband(
            self._model_builder,
            objective="val_loss",
            max_epochs=30,
            factor=3,
            directory=TUNER_PROJECT_NAME_3L,
            project_name="trials",
        )
        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        self.tuner.search(
            X, y, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=True
        )
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=3)[0]

    def _get_best_epoch(self, X, y):
        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = self.tuner.hypermodel.build(self.best_hps)
        history = model.fit(X, y, epochs=50, validation_split=0.2)

        val_per_epoch = history.history["val_loss"]
        self.best_epoch = val_per_epoch.index(min(val_per_epoch)) + 1
        print("Best epoch: %d" % (self.best_epoch,))

    def _final_train(self, X, y):
        self.hypermodel = self.tuner.hypermodel.build(self.best_hps)

        # Retrain the model
        self.hypermodel.fit(X, y, epochs=self.best_epoch, validation_split=0.2)

    def fit(self):
        self._search(self.X, self.y)
        self._get_best_epoch(self.X, self.y)
        self._final_train(self.X, self.y)
        self.hypermodel.save(os.path.join(TUNER_PROJECT_NAME_3L, TUNER_MODEL_FOLDER_3L))

    def export_model(self):
        return self.hypermodel

mdl = TunerClassifier(X, y)
mdl.fit()

model = mdl.export_model()

joblib.dump(model, "../../checkpoints/model.joblib")

shutil.rmtree(TUNER_MODEL_FOLDER_3L)