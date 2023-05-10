import numpy as np
from keras.layers import Dense                                  #Dense - это класс односвязного/линейного слоя, все нейроны связаны друг с другом.
from keras.models import Sequential                             #Sequental - Sequential - это класс последовательности слоев в нейронной сети,  у нас пока будет только один слой, но сюда можно добавлять сколь угодно слоев и это будет сеть,  состоящая из последовательности слоев.
import tensorflow as tf
from keras.activations import linear                            #Linear - импорт линейной функции активации
import pandas as pd

#Создание обучающих данных
x = np.array([[1], [-3], [2], [5], [4], [7], [8]])
y = np.array([4, -12, 8, 20, 16, 28, 32])


model = Sequential([Dense(1, input_shape=(1,), activation='linear')])


#linear(w1 * X[:1] + w0)
#w1 * X[:1] + w0 = результат. В данном случае w1 - вес, x1 - вводное число, w0 -bias(отклонения)
model.summary()
w1, w0 = model.get_weights()
print(w1, w0)

#На выходе есть два параметра, первый - параметр на обучение, второй - bias(отклонение)

model.compile(optimizer='sgd', loss='mse', metrics='mae') 
#optimizer - способ оптимизации. В данном случае выбран стохастический градиентный спуск
#loss - это то, что мы оптимизируем. В данном случае мы оптимизируем функции потери mse
#mettics(оцпионально) - передача метрик, которые мы высчитываем

model.fit(x, y, epochs=100)   #х, y - входные данные

#После 1 запуска функция потери mse составила 0.0095. Функция ошибки маленькая, результат удовлетворительный


#В этой части кода идёт проверка работспособности нейросети на новых данных
user_inp1, user_inp2 = 5, -9
print(f"Проверка на новых данных: {user_inp1} {user_inp2}")
print("Предсказание нейронной сети: ")
print(model.predict(np.array([[user_inp1], [user_inp2]])))

#В итоге нейронная сеть вывела 2 значения [[ 15.038922], [-26.672468]], что очень близко к правде

#Команда вывода весов до обучения и после обучения
nw1, nw0 = model.get_weights()
print('w1 before', w1, 'w1 after', nw1)
print('w0 before', w0, 'w0 after', nw0)


# Выходные данные
#w1 before [[0.04972327]] w1 after [[2.964854]]
#w0 before [0.] w0 after [0.2420863]

# В результате обучение w1 приблизилось к тройке, а w0 стал отличным от нуля, что доказывает обучение нейронной сети
