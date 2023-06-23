import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Input,\
    Dense, LeakyReLU, BatchNormalization, Reshape, Conv2DTranspose, Conv2D, MaxPooling2D, Flatten
from random import sample
import matplotlib.pyplot as plt

def train_gan(images, generator, discriminator, combined_model, epochs=100, batch_size=10):
    batches_num = int(images.shape[0] / batch_size)
    loss_fake = []
    loss_real = []
    acc_fake = []
    acc_real = []

    for epoch in range(epochs):
        print("Current epoch: ", epoch)
        images_left = set([i for i in range(images.shape[0])])

        for batch in range(batches_num):
            batch_images_indexes = sample(images_left, batch_size)
            images_left = images_left - set(batch_images_indexes)

            fake_images = generator.predict(tf.random.normal([batch_size, 10]))
            real_images = images[batch_images_indexes]

            fake_images_loss = discriminator.train_on_batch(fake_images, np.zeros(batch_size))
            real_images_loss = discriminator.train_on_batch(real_images, np.ones(batch_size))
            #loss_fake.append(fake_images_loss[0])
            acc_fake.append(fake_images_loss[1])
            #loss_real.append(real_images_loss[0])
            #acc_real.append(real_images_loss[1])

            combined_model.train_on_batch(tf.random.normal([batch_size, 10]), np.ones(batch_size))

        if epoch % 10 == 0:

            example_image = generator.predict(tf.random.normal([1, 10]))[0]
            mini = np.min(example_image)
            maxi = np.max(example_image.max())
            example_image = (example_image - mini) / (maxi - mini) * 255
            cv2.imwrite("sample_more_filters_epoch_" + str(epoch) + ".png", example_image.astype("uint8"))

    #epochs_list = [i for i in range(0, epochs)]
    plt.plot(acc_fake)
    plt.savefig("acc_fake.png")
    return generator

generator = tf.keras.Sequential([
    InputLayer(10),

    Dense(50),
    LeakyReLU(0.1),
    BatchNormalization(momentum=0.8),

    Dense(25 * 6),
    LeakyReLU(alpha=0.2),
    BatchNormalization(momentum=0.8),

    Reshape((6, 25, 1)),

    Conv2DTranspose(100, [3, 3], strides=[2, 2], padding="same"),
    BatchNormalization(momentum=0.8),
    LeakyReLU(alpha=0.2),

    Conv2DTranspose(50, [3, 3], strides=[2, 2], padding="same"),
    BatchNormalization(momentum=0.8),
    LeakyReLU(alpha=0.2),

    Conv2DTranspose(1, [3, 3], strides=[2, 2], padding="same", activation="tanh")
    ])
generator.compile(loss='binary_crossentropy', optimizer="adam")
#generator.summary()

discriminator_cnn = tf.keras.Sequential([
    InputLayer([48, 200, 1]),

    Conv2D(32, [5, 5], activation="relu"),
    MaxPooling2D([2, 2]),

    Conv2D(64, [5, 5], activation="relu"),
    MaxPooling2D([2, 2]),

    Conv2D(128, [5, 5], activation="relu"),

    Flatten(),

    Dense(100),
    Dense(10),
    Dense(1, activation="sigmoid")
])
#discriminator_cnn.summary()

discriminator_cnn.compile(loss="binary_crossentropy",
                          optimizer="adam", metrics=["accuracy"])
discriminator_cnn.trainable = False

model_input = Input(shape=(10,))
gen_image = generator(model_input)
check = discriminator_cnn(gen_image)

combined_model = tf.keras.Model(model_input, check)
combined_model.compile(loss='binary_crossentropy', optimizer="adam")

images_path = os.path.join(os.getcwd(), "samples")
images_names = os.listdir(images_path)
loaded_images = np.array([cv2.imread(os.path.join(images_path, name))[0:48, 0:200, 0:1] for name in images_names], dtype="float64")
loaded_images = (loaded_images / 127.0) - 1

trained_generator = train_gan(loaded_images, generator, discriminator_cnn, combined_model, epochs=100, batch_size=30)
trained_generator.save("model\\savedmodel")