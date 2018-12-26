import numpy as np
#import matplotlib.pyplot as plt
import os, shutil
from glob import glob
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers.core import Activation
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.model_selection import train_test_split


def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))
  
  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')
  
  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()


def create_set_folder(set_dir, nevus, melanoma, seborrheic):
    os.mkdir(set_dir)
    
    os.mkdir(set_dir + '/nevus')
    for f in nevus:
        shutil.copy(f, set_dir + '/nevus')
    
    os.mkdir(set_dir + '/melanoma')
    for f in melanoma:
        shutil.copy(f, set_dir + '/melanoma')

    os.mkdir(set_dir + '/seborrheic')
    for f in seborrheic:
        shutil.copy(f, set_dir + '/seborrheic')


print("Creating train and test datasets...")

nevus = glob('dataset/nevus/*.jpg')
melanoma = glob('dataset/melanoma/*.jpg')
seborrheic = glob('dataset/seborrheic/*.jpg')

nevus_train, nevus_test = train_test_split(nevus, test_size=0.30)
melanoma_train, melanoma_test = train_test_split(melanoma, test_size=0.30)
seborrheic_train, seborrheic_test = train_test_split(seborrheic, test_size=0.30)

create_set_folder('train', nevus_train, melanoma_train, seborrheic_train)
create_set_folder('test', nevus_test, melanoma_test, seborrheic_test)


# Plot some samples of each class
plotsamples = False

if plotsamples:
    n = np.random.choice(nevus_train, 5)
    m = np.random.choice(melanoma_train, 5)
    s = np.random.choice(seborrheic_train, 5)

    data = np.concatenate((n, m, s))
    labels = 5 * ['Normal'] + 5 * ['Melanoma'] + 5 * ['Seborrheic']

    N, R, C = 15, 3, 5
    plt.figure(figsize=(12, 9))
    for k, (src, label) in enumerate(zip(data, labels)):
        im = Image.open(src).convert('RGB')
        plt.subplot(R, C, k + 1)
        plt.title(label)
        plt.imshow(np.asarray(im))
        plt.axis('off')
    plt.show()


CLASSES = 3
WIDTH = 299
HEIGHT = 299
BATCH_SIZE = 32

print("Building model...")

# Setup model
K.set_image_data_format('channels_last')
model = Sequential()

## SINGLELAYER ##

model.add(Conv2D(40, kernel_size=5, padding="same", input_shape=(WIDTH, HEIGHT, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=CLASSES, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## MULTILAYER ##

# Layer 1
#model.add(Conv2D(40, kernel_size=5, padding="same", input_shape=(WIDTH, HEIGTH, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 2
#model.add(Conv2D(70, kernel_size=3, padding="same", activation = 'relu'))
#model.add(Conv2D(500, kernel_size=3, padding="same", activation = 'relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#model.add(Conv2D(1024, kernel_size=3, padding="valid", activation = 'relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 3 - Flatten layer
#model.add(Flatten())
#model.add(Dense(units=100, activation='relu'))
#model.add(Dropout(0.1))
#model.add(Dense(units=100, activation='relu'))
#model.add(Dropout(0.1))
#model.add(Dense(units=100, activation='relu'))
#model.add(Dropout(0.3))

#model.add(Dense(CLASSES))
#model.add(Activation("softmax"))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# data prep
train_datagen = ImageDataGenerator(
    #preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    #preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(HEIGHT, WIDTH),
		batch_size=BATCH_SIZE,
		class_mode='categorical')
    
validation_generator = validation_datagen.flow_from_directory(
    'test',
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

EPOCHS = 5
BATCH_SIZE = 32
STEPS_PER_EPOCH = 320
VALIDATION_STEPS = 64

MODEL_FILE = 'filename.model'

print("Training model...")

history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS)
  
model.save(MODEL_FILE)

plot_training(history)


# remove test and train paths
shutil.rmtree('/train') 
shutil.rmtree('/test') 
