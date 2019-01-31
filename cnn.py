import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, shutil
from glob import glob
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.layers.core import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import Callback
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def recall_metric(y_true, y_pred):
    """Recall metric.
 
   Only computes a batch-wise average of recall.
 
   Computes the recall, a metric for multi-label classification of
   how many relevant items are selected.
   """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
 
 
def precision_metric(y_true, y_pred):
    """Precision metric.
 
   Only computes a batch-wise average of precision.
 
   Computes the precision, a metric for multi-label classification of
   how many selected items are relevant.
   """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
 
 
def f1_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def plot_training(history, model_name):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, label="Training accuracy")
    plt.plot(epochs, val_acc, label="Validation accuracy")
    plt.legend(loc='best')
    plt.title('Training and validation accuracy')
    plt.savefig(model_name + '_acc.png')
    plt.close()
  
    plt.figure()
    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.legend(loc='best')
    plt.title('Training and validation loss')
    plt.savefig(model_name + '_loss.png')
    plt.close()


def create_set_folder(set_dir, nevus, melanoma, seborrheic):
    # remove test and train paths
    if os.path.exists(set_dir):
        shutil.rmtree(set_dir) 
    
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


def equalize_set_lengths(set1, set2, set3):
    # Compute minimum length of all sets
    min_length = min(map(len, [set1, set2, set3]))
    return set1[:min_length], set2[:min_length], set3[:min_length]


def create_sets():
    print("Creating train and test datasets...")

    nevus_train = glob('dataset/nevus/*.jpg')
    melanoma_train = glob('dataset/melanoma/*.jpg')
    seborrheic_train = glob('dataset/seborrheic/*.jpg')

    nevus_val = glob('valset/nevus/*.jpg')
    melanoma_val = glob('valset/melanoma/*.jpg')
    seborrheic_val = glob('valset/seborrheic/*.jpg')

    nevus_test = glob('testset/nevus/*.jpg')
    melanoma_test = glob('testset/melanoma/*.jpg')
    seborrheic_test = glob('testset/seborrheic/*.jpg')

    # Equalize set lengths
    nevus_train, melanoma_train, seborrheic_train = \
            equalize_set_lengths(nevus_train, melanoma_train, seborrheic_train)
    nevus_val, melanoma_val, seborrheic_val = \
            equalize_set_lengths(nevus_val, melanoma_val, seborrheic_val)
    nevus_test, melanoma_test, seborrheic_test = \
            equalize_set_lengths(nevus_test, melanoma_test, seborrheic_test)

    create_set_folder('train', nevus_train, melanoma_train, seborrheic_train)
    create_set_folder('val', nevus_val, melanoma_val, seborrheic_val)
    create_set_folder('test', nevus_test, melanoma_test, seborrheic_test)


def normalize_data(x):
    return (x - K.mean(x)) / K.std(x)


def create_model(CLASSES=3, WIDTH=128, HEIGHT=128, BATCH_SIZE=32):
    print("Building model...")

    # Setup model
    K.set_image_data_format('channels_last')
    model = Sequential()

    ## NEW MULTILAYER ##
    # Current dimension: 128, 128, 3
    # Layer 1
    model.add(Activation(activation=normalize_data, input_shape=(WIDTH, HEIGHT, 3)))

    # Layer 2
    model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.2))
    # New dimension: 31, 31, 32

    # Layer 3
    model.add(Conv2D(64, kernel_size=3, padding="same", activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # New dimension: 14, 14, 64

    # Layer 4
    model.add(Conv2D(128, kernel_size=3, padding="same", activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # New dimension: 7, 7, 128
    
    # Layer 4
    #model.add(Conv2D(256, kernel_size=3, padding="same", activation='relu'))
    #model.add(BatchNormalization(axis=-1))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.1))
    # New dimension: 3, 3, 256

    # Layer 5 - Flatten layer
    model.add(Flatten())
    model.add(Dense(units=4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    #model.add(Dense(units=4096, activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.1))
    model.add(Dense(units=1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(units=CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
            metrics=['accuracy', precision_metric, recall_metric, f1_metric])

    return model


def compile_and_train(model, model_name, WIDTH=128, HEIGHT=128, 
        BATCH_SIZE=32, EPOCHS=5):
    # data prep
    train_datagen = ImageDataGenerator(
        #preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(
        #preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
        
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
        
    validation_generator = validation_datagen.flow_from_directory(
        'data/val',
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    MODEL_FILE = model_name + '.model'
    STEPS_PER_EPOCH = train_generator.samples // BATCH_SIZE * 4
    VALIDATION_STEPS = validation_generator.samples // BATCH_SIZE * 2
    class_weights = class_weight.compute_class_weight(
            'balanced', np.unique(train_generator.classes), train_generator.classes)

    print("Training model...")

    hst = AccuracyHistory()
    history = model.fit_generator(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_generator,
        validation_steps=VALIDATION_STEPS,
        callbacks=[hst],
        class_weight=class_weights)
      
    model.save(MODEL_FILE)

    plot_training(history, model_name)

    with open(model_name + '_hist.dat', 'w') as f:
        for e in hst.acc:
            f.write(str(e) + " ")


def binary_metrics(conf_matrix, change_class, max_class):
    tp = conf_matrix[0:change_class, 0:change_class].sum()
    fn = conf_matrix[change_class:max_class+1, 0:change_class].sum()
    fp = conf_matrix[0:change_class, change_class:max_class+1,].sum()
    tn = conf_matrix[change_class:max_class+1, change_class:max_class+1].sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    f1score = (2 * precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1score


def evaluate_model(model, WIDTH=128, HEIGHT=128, BATCH_SIZE=32, binary=False):
    test_datagen = ImageDataGenerator(
        #preprocessing_function=preprocess_input,
        #rotation_range=40,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #vertical_flip=True,
        #fill_mode='nearest'
        )
    
    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    
    STEPS_PER_EPOCH = np.math.ceil(test_generator.samples / BATCH_SIZE)
    
    pred = model.predict_generator(test_generator, steps=STEPS_PER_EPOCH)
    y_pred = np.argmax(pred, axis=1)
    y_true = test_generator.classes
    cm = confusion_matrix(y_true, y_pred)
    print(test_generator.class_indices)

    # Print Binary results
    print("Test results: ")
    print(cm)
    if binary:
        print("Binary results: ")
        print(binary_metrics(cm, 1, 1))

    # Print 3 classes results
    print("Multiclass results: ")
    precision, recall, f1score, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted')
    print(accuracy_score(y_true, y_pred), precision, recall, f1score)


if __name__== "__main__":
    EPOCHS = 50
    BATCH_SIZE = 32
    WIDTH = 128 
    HEIGHT = 128
    
    #create_sets()
    model = create_model(CLASSES=2)
    compile_and_train(
            model, 
            'cnn/binary01',
            WIDTH=WIDTH, 
            HEIGHT=HEIGHT,
            BATCH_SIZE=BATCH_SIZE,
            EPOCHS=EPOCHS
    )
    evaluate_model(model, WIDTH=WIDTH, HEIGHT=HEIGHT, BATCH_SIZE=BATCH_SIZE)

    # remove test and train paths
    #shutil.rmtree('train')
    #shutil.rmtree('val')
    #shutil.rmtree('test')

