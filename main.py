import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
model.compile(optimizer="adam",loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10)

model.save("handwritten_recognition.keras")

#

loss,acc = model.evaluate(x_test,y_test)
print(loss,acc)

#model = tf.keras.models.load_model('handwritten_recognition.keras')
image_number = 0
while os.path.isfile(f"digits/digit_number{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit_number{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(prediction)
        print(f'the digit is probably {np.argmax(prediction)}')
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image_number += 1
