import tensorflow as tf
from keras.layers import Dense              # Dense - это класс односвязного/линейного слоя, все нейроны связаны друг с другом.
from keras.models import Sequential         # Sequental - Sequential - это класс последовательности слоев в нейронной сети, 
"""Sequential - это класс последовательности слоев в нейронной сети, 
у нас пока будет только один слой, но сюда можно добавлять сколь угодно слоев и это будет сеть, 
состоящая из последовательности слоев.
Последовательность означает, что вход, поступивший в нейросеть,идёт последовательно по всей сети, нигде не зацикливаясь"""

model = Sequential([Dense(1, input_shape=(1,), activation='relu')])

"""
Dense - это класс односвязного/линейного слоя, все нейроны связаны друг с другом.
Dense(1 - в сети 1 нейрон

units=1 - это количество нейронов в слое. У нас 1 нейрон.

input_shape=(1,) - это входная размерность объекта. У нас только 1 вход.

activation='key' - это функция активации, которая добавляет в слой нелинейности, 
именно из-за неё мы можем получать
более сложные результаты работы сети."""

print(model.get_weights())             #команда вывода весов. param - обучающиеся веса
model.summary()                        #Вывод "резюме" модели

