import csv
import tensorflow as tf
import random
import pandas as pd
import numpy as np

# Uamos panda para leer el archivo csv
ipd = pd.read_csv("iris.csv")

# Tomamos las especies, las transformamos en una matriz con 3 columnas, correspondientes a cada tipo con valor booleano
# y la insertamos en la columna One-hot
species = list(ipd['Species'].unique())
ipd['One-hot'] = ipd['Species'].map(lambda x: np.eye(len(species))[species.index(x)] )

# Desordenamos las muestras, y dejamos 100 para entrenamiento y 50 para test
shuffled = ipd.sample(frac=1)
trainingSet = shuffled[0:len(shuffled)-50]
testSet = shuffled[len(shuffled)-50:]

# Definimos el feed de muestra, los pesos y las bias
inp = tf.placeholder(tf.float32, [None, 4])
weights = tf.Variable(tf.zeros([4, 3]))
bias = tf.Variable(tf.zeros([3]))

# Realizamos la activacion con softmax
y = tf.nn.softmax(tf.matmul(inp, weights) + bias)

# Definimos el feed con los resultados esperados
y_ = tf.placeholder(tf.float32, [None, 3])
# Calculamos la entropia cruzada
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# Entrenamos la red utilizando el algoritmo de optimizacion de gradiente descendente
# llamado Adaptive Moment Estimation (Adam)
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
# Define si es una prediccion correcta evaluando si la posicion del valor mas
# grande de la muestra, es el mismo que el calculado
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# Precision
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

keys = ['Sepal Length', 'Sepal Width','Petal Length', 'Petal Width']
# Iteramos el entrenamiento 1000 veces
for i in range(1000):
    train = trainingSet.sample(50)
    sess.run(train_step, feed_dict={inp: [x for x in train[keys].values],
                                    y_: [x for x in train['One-hot'].as_matrix()]})


print sess.run(accuracy, feed_dict={inp: [x for x in testSet[keys].values],
                                    y_: [x for x in testSet['One-hot'].values]})
# Creamos un metodo predictivo
def classify(inpv):
    dim = y.get_shape().as_list()[1]
    res = np.zeros(dim)
    largest = sess.run(tf.argmax(y,1), feed_dict={inp: inpv})[0]
    return np.eye(dim)[largest]

# Tomamos una muestra y testeamos
sample = shuffled.sample(1)
print "Classified as %s" % classify(sample[keys])
