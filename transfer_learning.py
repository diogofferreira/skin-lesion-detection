import numpy as np
import matplotlib.pyplot as plt
import os, shutil
from glob import glob
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
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
    plt.savefig('history_acc.png')
    
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.savefig('history_loss.png')
    #plt.show()


def create_set_folder(set_dir, nevus, melanoma, seborrheic):
    os.mkdir(set_dir)
    
    os.mkdir(set_dir + '/nevus_seborr')
    for f in nevus:
        shutil.copy(f, set_dir + '/nevus_seborr')
    
    for f in seborrheic:
        shutil.copy(f, set_dir + '/nevus_seborr')
    
    os.mkdir(set_dir + '/melanoma')
    for f in melanoma:
        shutil.copy(f, set_dir + '/melanoma')

    #os.mkdir(set_dir + '/seborrheic')
    #for f in seborrheic:
    #    shutil.copy(f, set_dir + '/seborrheic')


nevus = glob('dataset/nevus/*.jpg')
melanoma = glob('dataset/melanoma/*.jpg')
seborrheic = glob('dataset/seborrheic/*.jpg')

nevus_train, nevus_test = train_test_split(nevus, test_size=0.30)
melanoma_train, melanoma_test = train_test_split(melanoma, test_size=0.30)
seborrheic_train, seborrheic_test = train_test_split(seborrheic, test_size=0.30)

create_set_folder('train', nevus_train, melanoma_train, seborrheic_train)
create_set_folder('test', nevus_test, melanoma_test, seborrheic_test)


"""
# Plot some samples of each class

n = np.random.choice(nevus_train, 5)
m = np.random.choice(melanoma_train, 5)
s = np.random.choice(seborrheic_train, 5)

data = np.concatenate((n, m, s))
labels = 5 * ['Nevus'] + 5 * ['Melanoma'] + 5 * ['Seborrheic']

N, R, C = 15, 3, 5
plt.figure(figsize=(12, 9))
for k, (src, label) in enumerate(zip(data, labels)):
    im = Image.open(src).convert('RGB')
    plt.subplot(R, C, k + 1)
    plt.title(label)
    plt.imshow(np.asarray(im))
    plt.axis('off')
plt.show()
"""

CLASSES = 2

# Setup model
inception_base_model = InceptionV3(weights='imagenet', include_top=False)
vgg_base_model = VGG19(weights='imagenet', include_top=False)
resnet_base_model = ResNet50(weights='imagenet', include_top=False)

x = inception_base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.4)(x)
predictions_inception = Dense(CLASSES, activation='softmax')(x)

x = vgg_base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.4)(x)
predictions_vgg = Dense(CLASSES, activation='softmax')(x)

x = resnet_base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.4)(x)
predictions_resnet = Dense(CLASSES, activation='softmax')(x)

inception_model = Model(inputs=inception_base_model.input, outputs=predictions_inception)
vgg_model = Model(inputs=vgg_base_model.input, outputs=predictions_vgg)
resnet_model = Model(inputs=resnet_base_model.input, outputs=predictions_resnet)
   
# transfer learning
for layer in inception_base_model.layers:
    layer.trainable = False

for layer in vgg_base_model.layers:
    layer.trainable = False

for layer in resnet_base_model.layers:
    layer.trainable = False

# Optimizer
sgd = optimizers.SGD(nesterov=True)

inception_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
vgg_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
resnet_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

WIDTH = 224
HEIGHT = 224
BATCH_SIZE = 32

# data prep
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

train_generator_inception = train_datagen.flow_from_directory(
    'train',
    target_size=(299, 299),
    batch_size=BATCH_SIZE,
    class_mode='categorical')
    
validation_generator_inception = validation_datagen.flow_from_directory(
    'test',
    target_size=(299, 299),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

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

# Inception training
history = inception_model.fit_generator(
    train_generator_inception,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator_inception,
    validation_steps=VALIDATION_STEPS)
  
inception_model.save('inception.model')

plot_training(history)

# VGG training
history = vgg_model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS)
  
vgg_model.save('vgg.model')

plot_training(history)

# ResNet training
history = resnet_model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS)
  
inception_model.save('resnet.model')

plot_training(history)

# remove test and train paths
shutil.rmtree('train') 
shutil.rmtree('test') 
