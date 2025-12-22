import cv2
import numpy as np
import tensorflow as tf
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator

train_image_generator = ImageDataGenerator(rescale=1./255,
                                           zoom_range=0.2,
                                           rotation_range=10)

images_train = train_image_generator.flow_from_directory(directory="../dataset/train",
                                                         target_size=(128, 128),
                                                         batch_size=4,
                                                         shuffle=True)

test_image_generator = ImageDataGenerator(rescale=1./255)

images_test = test_image_generator.flow_from_directory(directory="../dataset/test",
                                                       target_size=(128, 128),
                                                       batch_size=1,
                                                       shuffle=False)

def create_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))

    model.summary()

    model.compile("Adam", loss='binary_crossentropy', metrics=["Accuracy"])
    model.fit(images_train, validation_data=images_test, epochs=10, batch_size=128)
    model.save("./weights/weightV1.h5")

model = tf.keras.models.load_model("weights/weightV1.h5")

imagem = cv2.imread("../dataset/test/Normal/0828.JPG")

imagem = cv2.resize(imagem, (128, 128))
imagem = imagem / 255
imagem = imagem.reshape(-1, 128, 128, 3)
y_predict = model.predict(imagem)
y_predict = np.argmax(y_predict)
print(images_test.class_indices)
print(y_predict)
