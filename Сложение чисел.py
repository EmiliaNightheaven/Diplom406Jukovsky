import numpy as np
import tensorflow as tf
from keras.activations import linear

#Создание случайных чисел
X1 = np.random.randint(1, 10, size=50)
X2 = np.random.randint(1, 10, size=50)

y = X1 + X2                    #Сложение случайных чисел
X = np.vstack([X1, X2]).T      #Запись чисел в массив
#print(X)                      #Вывод массива

y = y[None]
y = y.T
#print(y)                      #Вывод результата сложения

#Для улучшения обучения нейронной сети, для уменьшения разброса данные нормализуют (в этом случае данные будут разбросаны от 0 до 1)
from sklearn.preprocessing import MinMaxScaler          #Библиотека для нормализации
mms = MinMaxScaler()                                    #Скалированние
X_norm = mms.fit_transform(X)                           #Нормирование данных

#print(X_norm)  
# В данной нейросети будет два слоя, два входных значения, одно выходное значение
# В первом слое будет  3 нейрона, во 2 слое 1 нейрон                                

from keras.layers import Dense
from keras.models import Sequential                     
# Используется Sequential. У нас слои последовательные, сначала 2 входа идут в 3 нейрона, 
# а затем выходы из этих 3 нейронов идут в один нейрон.                     
tf.random.set_seed(9)

model = Sequential([
    #Первый слой - полносвязный, на вход принимает два значения (input_shape),
    # в самом слое 2 нейрона, активация линейная. После этого данные поступают в следующий слой
    Dense(3, input_shape=(2,), activation='linear'),    
 
    #В этом слою 1 нейрон, нет входных данных, активация линейная
    Dense(1, activation='linear')
])

model.summary()
print(model.get_weights())

#Резюме и параметры нейросети

# Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 3)                 9                    

#  dense_1 (Dense)             (None, 1)                 4

# =================================================================
# Total params: 13
# Trainable params: 13
# Non-trainable params: 0
# _________________________________________________________________

# Веса для 1 слоя
# [array([[ 1.0949442 ,  0.90116215,  0.15849483],
#        [-0.05964518, -0.7692275 , -0.2285415 ]], dtype=float32),              
# array([0., 0., 0.], dtype=float32), 
#Веса для второго слоя
# array([[-0.08689535],
#        [ 0.46066284],
#        [-0.22816741]], dtype=float32), array([0.], dtype=float32)]

#Количество весов для одного нейрона равно 2. На каждый нейрон поступает два входа(2x3=6), 
# а так же у каждого нейрона есть 1 bias(1x3), итого 9 весов 
# 
#Второй слой ждет на вход 3 сигнала, еще 1 bias.

model.compile(optimizer='sgd', loss='mse', metrics='mae')
# optimizer - sgd - сохастический градиент, способ оптимизации
# loss - то, что мы оптимизируем, в данном случае функцию потери 'mse'
# metrics(оцпионально) - передача метрик, которые мы высчитываем

model.fit(X_norm, y, epochs=200)

# X_norm - входные данные, y - выходные данные, epochs - кол-во циклов

#Проверка нейросети на других данных
test_X = [[4, 2],
          [6, 2]]
test_X = mms.transform(test_X)
print("Предсказание нейронной сети: ")
print(model.predict(np.array(test_X)))

#Вывод обучающих данных в таблицу
import pandas as pd

pd.DataFrame({
    'x1': X[:, 0],
    'x2': X[:, 1],
    'true': np.squeeze(y),
    'pred': np.squeeze(model.predict(X_norm))
}).head(10)

nw1, nw0 = model.get_weights()