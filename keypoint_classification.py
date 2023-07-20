"""plik służący do trenowania modelu na podstawie zaznaczonych przez użytkownika gestów i wpisanych do pliku keypoint.csv"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

# WAŻNE: Może być konieczna zmiana lokalizacji na globalną

# Lokalizacja plików z zapisanymi współrzędnymi punktów do nauki modelu dla jednej/dwóch rąk
# Plik utworzony poprzez naciskanie #k przy wykonywaniu żądanego gestu przy odpowiednim indeksie ustawionym na suwaku
# Kolejność zapisywania gestów i ich indeksy zgodnie z plikiem z nazwami keypoint_classifier_label.csv
DATASET_ONE_HAND = 'model/keypoint_classifier/keypoint.csv'
DATASET_TWO_HANDS = 'model2/keypoint_classifier2/keypoint2.csv'

# Lokalizacja do zapisania plików z modelem Tensorflow
MODEL_SAVE_PATH_ONE_HAND = 'model/keypoint_classifier/keypoint_classifier.hdf5'
MODEL_SAVE_PATH_TWO_HANDS = 'model2/keypoint_classifier2/keypoint_classifier2.hdf5'

# Lokalizacja do zapisania plików z modelem Tensorflow Lite
TFLITE_SAVE_PATH_ONE_HAND = 'model/keypoint_classifier/keypoint_classifier.tflite'
TFLITE_SAVE_PATH_TWO_HANDS = 'model2/keypoint_classifier2/keypoint_classifier2.tflite'

# Liczba rozpoznawanych gestów dla jednej/dwóch rąk, liczone od 1
# Jeżeli dodano nowy gest, należy zwiększyć te wartości o 1 za każdy nowy gest
NUM_CLASSES_ONE_HAND = 14
NUM_CLASSES_TWO_HANDS = 5

class Trainer:
    def __init__(
        self,
        numberOfHands,
    ):
        self.numberOfHands = numberOfHands
        if self.numberOfHands == 1:
            self.dataset = DATASET_ONE_HAND
            self.model_save_path = MODEL_SAVE_PATH_ONE_HAND
            self.model_TFlite_save_path = TFLITE_SAVE_PATH_ONE_HAND
            # ile jest zaplanowanych gestów do rozpoznawania
            self.number = NUM_CLASSES_ONE_HAND
        elif self.numberOfHands == 2:
            self.dataset = DATASET_TWO_HANDS
            self.model_save_path = MODEL_SAVE_PATH_TWO_HANDS
            self.model_TFlite_save_path = TFLITE_SAVE_PATH_TWO_HANDS
            # ile jest zaplanowanych gestów do rozpoznawania
            self.number = NUM_CLASSES_TWO_HANDS

    def training(
    self,
    ):
        X_dataset = np.loadtxt(self.dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))

        y_dataset = np.loadtxt(self.dataset, delimiter=',', dtype='int32', usecols=(0))

        X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75,
                                                            random_state=RANDOM_SEED)

        # ta klasa grupuje liniowe warstwy w model tf.keras.Model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input((21 * 2,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(self.number, activation='softmax')
        ])

        model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            self.model_save_path, verbose=1, save_weights_only=False)
        es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

        # konfigracja modelu do treningu
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        # trenuje model przez 1000 iteracji
        model.fit(
            X_train,
            y_train,
            epochs=1000,
            batch_size=128,
            validation_data=(X_test, y_test),
            callbacks=[cp_callback, es_callback]
        )

        val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)

        model = tf.keras.models.load_model(self.model_save_path)

        # zapisanie modelu
        model.save(self.model_save_path, include_optimizer=False)

        # konwersja do modelu Tensorflow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quantized_model = converter.convert()

        open(self.model_TFlite_save_path, 'wb').write(tflite_quantized_model)

        # wykonanie analizy na jednym z zebranych przykładów wykorzystanych do trenowania
        interpreter = tf.lite.Interpreter(model_path=self.model_TFlite_save_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))

        interpreter.invoke()
        tflite_results = interpreter.get_tensor(output_details[0]['index'])

        print(np.squeeze(tflite_results))
        print(np.argmax(np.squeeze(tflite_results)))

while True:
    print("Dla ilu rąk należy trenować model? Wpisz '1' albo '2'")
    numberOfHands = input()
    numberOfHands = int(numberOfHands)
    if numberOfHands == 1:
        trainer = Trainer(2)
        break
    elif numberOfHands == 2:
        trainer = Trainer(2)
        break

trainer.training()


