from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Se cargan los datasets, de entrenamiento y de testeo
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=IRIS_TRAINING, # Path del archivo csv
    target_dtype=np.int, # Tipo de dato de los objetivos especificado con Numpy
    features_dtype=np.float32) # Tipo de dato de las caracteristicas
                               # especificado con Numpy
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

# tf.contrib.layers.real_valued_column Define las columnas que representan
# las caracteristicas a analizar. El primer parametro es el nombre
# el segundo especifica la dimencion que debe contener las columnas de
# entrada, en este caso las 4 primeras
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Construir 3 capas DNN con 10, 20, 10 unidades respectivamente.
# DNN es Deep Neural Network Classifier, feature_columns son las columnas de
# entrada, lo que se conoce como los input, hidden_units es un array que
# representa las capas ocultas indicando cuantas neuronas tiene por capas
# en el ejemplo de abajo son 3 capas ocultas con 10, 20 y 10 neuronas cada
# una respectivamente y n_classes representa las clases de salidas en que
# representara la neurona de salida, model_dir es donde tensorflow guarda
# los checkpoints
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

# Ajustar el modelo (fit=ajustar).
classifier.fit(x=training_set.data, # Caracteristicas de entrenamiento
               y=training_set.target, # Valor objetivo real de entrenamiento
               steps=2000)

# Evaluar la exactitud (accuracy=exactitud)
accuracy_score = classifier.evaluate(x=test_set.data, # Caracteristicas de prueba
                                     y=test_set.target)["accuracy"] # Valor objetivo real de prueba
print('Accuracy: {0:f}'.format(accuracy_score))

# Clasificar dos flores de ejemplo
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
# classifier.predict predice segun el entrenamiento anterior y las pruebas
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))
