import numpy as np
import matplotlib.pyplot as plt
import os, shutil
from glob import glob
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def plot_training(history, model_name, save=True):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.figure()
    
    if save:
        plt.savefig(model_name + '_acc.png')
     
    plt.plot(epochs, loss, 'r*')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    
    if save:
        plt.savefig(model_name + '_loss.png')
    else:
        plt.show()


def create_set_folder(set_dir, nevus, melanoma, seborrheic):
    os.mkdir(set_dir)
    
    os.mkdir(set_dir + '/nevus')
    for f in nevus:
        shutil.copy(f, set_dir + '/nevus')
    
    #for f in seborrheic:
    #    shutil.copy(f, set_dir + '/nevus_seborr')
    
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


def prepare_sets():
    nevus = glob('dataset/nevus/*.jpg')
    melanoma = glob('dataset/melanoma/*.jpg')
    seborrheic = glob('dataset/seborrheic/*.jpg')

    # Split train set from the others
    nevus_train, nevus_test = train_test_split(nevus, test_size=0.40)
    melanoma_train, melanoma_test = train_test_split(melanoma, test_size=0.40)
    seborrheic_train, seborrheic_test = train_test_split(seborrheic, test_size=0.40)
    
    # Split test and cross validation set
    nevus_test, nevus_cross_val = train_test_split(nevus_test, test_size=0.50)
    melanoma_test, melanoma_cross_val = train_test_split(melanoma_test, test_size=0.50)
    seborrheic_test, seborrheic_cross_val = train_test_split(seborrheic_test, test_size=0.50)
    
    
    # Equalize set lengths
    nevus_train, melanoma_train, seborrheic_train = equalize_set_lengths(nevus_train, 
            melanoma_train, seborrheic_train)
    nevus_cross_val, melanoma_cross_val, seborrheic_cross_val = equalize_set_lengths(nevus_cross_val, 
            melanoma_cross_val, seborrheic_cross_val)
    nevus_test, melanoma_test, seborrheic_test = equalize_set_lengths(nevus_test, 
            melanoma_test, seborrheic_test)
    
    # Create folders with each set
    create_set_folder('train', nevus_train, melanoma_train, seborrheic_train)
    create_set_folder('cross_val', nevus_cross_val, melanoma_cross_val, seborrheic_cross_val)
    create_set_folder('test', nevus_test, melanoma_test, seborrheic_test)


def plot_data():
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


def prepare_models(CLASSES=3):
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

    # transfer learning
    for layer in inception_base_model.layers:
        layer.trainable = False

    for layer in vgg_base_model.layers:
        layer.trainable = False

    for layer in resnet_base_model.layers:
        layer.trainable = False
    
    inception_model = Model(inputs=inception_base_model.input, outputs=predictions_inception)
    vgg_model = Model(inputs=vgg_base_model.input, outputs=predictions_vgg)
    resnet_model = Model(inputs=resnet_base_model.input, outputs=predictions_resnet)
       
    return inception_model, vgg_model, resnet_model


def compile_and_train(model, model_name, WIDTH=224, HEIGHT=224, BATCH_SIZE=32):
    
    # Optimizer
    sgd = optimizers.SGD(nesterov=True)
    
    # Compile model
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    
    # data prep
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        #rotation_range=50,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #shear_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
        #rotation_range=50,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #vertical_flip=True,
        #fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
        
    validation_generator = validation_datagen.flow_from_directory(
        'cross_val',
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    EPOCHS = 20
    BATCH_SIZE = 32
    STEPS_PER_EPOCH = train_generator.samples // BATCH_SIZE * 5
    VALIDATION_STEPS = validation_generator.samples // BATCH_SIZE 

    # Inception training
    history = model.fit_generator(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_generator,
        validation_steps=VALIDATION_STEPS)
  
    model.save(model_name + '.model')

    plot_training(history, model_name)

    return model


def ensemble(models):
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    
    model = Model(model_input, y, name='ensemble')
    
    return model


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


def evaluate_model(model, HEIGHT=224, WIDTH=224, BATCH_SIZE=32, to_binary=False):
    test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        'test',
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    
    steps_per_epoch = np.math.ceil(test_generator.samples 
            / test_generator.batch_size)
    
    pred = model.predict_generator(test_generator,
            steps=steps_per_epoch, verbose=1)

    y_pred = np.argmax(pred, axis=1)
    y_true = test_generator.classes
    
    if to_binary:
        cm = confusion_matrix(y_true, y_pred)
        return binary_metrics(cm, 1, 2)

    precision, recall, f1score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    return accuracy_score(y_true, y_pred), precision, recall, f1score


if __name__== "__main__":
    
    prepare_sets()
    
    inception, vgg, resnet = prepare_models()
    
    #inception = compile_and_train(inception, 'inception', WIDTH=299, HEIGHT=299)
    vgg = compile_and_train(vgg, 'vgg')
    resnet = compile_and_train(resnet, 'resnet')
   
    """
    print('Inception 2 classes')
    inception = load_model('models/tl/2_classes/inception.model')
    print(evaluate_model(inception, HEIGHT=299, WIDTH=299))
    
    print('Inception 2 classes')
    vgg = load_model('models/tl/2_classes/vgg.model')
    print(evaluate_model(vgg))
    
    print('ResNet 2 classes')
    resnet = load_model('models/tl/2_classes/resnet.model')
    print(evaluate_model(resnet))
    """ 
    
    #print('Inception 3 classes')
    #inception = load_model('models/tl/3_classes/inception.model')
    #print(evaluate_model(inception, HEIGHT=299, WIDTH=299, to_binary=True))

    print('VGG 3 classes')
    #vgg = load_model('models/tl/3_classes/vgg.model')
    print(evaluate_model(vgg, to_binary=True))
    
    print('ResNet 3 classes')
    #resnet = load_model('models/tl/3_classes/resnet.model')
    print(evaluate_model(resnet, to_binary=True))

    #models = [vgg, resnet]

    # remove test and train paths
    shutil.rmtree('train') 
    shutil.rmtree('test') 
