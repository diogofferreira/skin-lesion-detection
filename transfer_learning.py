from PIL import Image
from glob import glob
from keras import backend as K
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Average, Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os, shutil
matplotlib.use('Agg')


def recall_metric(y_true, y_pred):
    # Compute recall metric based on predicted and true labels

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_metric(y_true, y_pred):
    # Compute recall metric based on predicted and true labels
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_metric(y_true, y_pred):
    # Compute recall metric based on predicted and true labels
    
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def save_plot(history, model_name):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    # Plot training and validation accuracy, over the epochs
    plt.figure()
    plt.plot(epochs, acc, label="Training accuracy")
    plt.plot(epochs, val_acc, label="Validation accuracy")
    plt.legend(loc='best')
    plt.title('Training and validation accuracy')
    plt.savefig(model_name + '_acc.png')
    plt.close()
  
    # Plot training and validation losso, over the epochs
    plt.figure()
    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.legend(loc='best')
    plt.title('Training and validation loss')
    plt.savefig(model_name + '_loss.png')
    plt.close()


def create_set_folder(set_dir, nevus, melanoma, seborrheic):
    
    # Create parent set folder
    os.mkdir(set_dir)
    
    # Create nevus set folder
    os.mkdir(set_dir + '/nevus')
    for f in nevus:
        shutil.copy(f, set_dir + '/nevus')
    
    # Create melanoma set folder
    os.mkdir(set_dir + '/melanoma')
    for f in melanoma:
        shutil.copy(f, set_dir + '/melanoma')

    # Create seborrheic set folder
    os.mkdir(set_dir + '/seborrheic')
    for f in seborrheic:
        shutil.copy(f, set_dir + '/seborrheic')


def equalize_set_lengths(set1, set2, set3):
    # Compute minimum length of all sets
    min_length = min(map(len, [set1, set2, set3]))
    
    return set1[:min_length], set2[:min_length], set3[:min_length]


def prepare_sets(equalize=False):
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
    
    if equalize:
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


def prepare_models(model_input, CLASSES=2):
    # Setup model
    inception_base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=model_input)
    vgg_base_model = VGG19(weights='imagenet', include_top=False, input_tensor=model_input)
    resnet_base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=model_input)

    # Change inception output layer
    x = inception_base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.4)(x)
    predictions_inception = Dense(CLASSES, activation='softmax')(x)

    # Change vgg output layer
    x = vgg_base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.4)(x)
    predictions_vgg = Dense(CLASSES, activation='softmax')(x)

    # Change resnet output layer
    x = resnet_base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.4)(x)
    predictions_resnet = Dense(CLASSES, activation='softmax')(x)

    # Freeze hidden layers

    for layer in inception_base_model.layers:
        layer.trainable = False

    for layer in vgg_base_model.layers:
        layer.trainable = False

    for layer in resnet_base_model.layers:
        layer.trainable = False
    
    # Create final TL model
    inception_model = Model(inputs=model_input, outputs=predictions_inception)
    vgg_model = Model(inputs=model_input, outputs=predictions_vgg)
    resnet_model = Model(inputs=model_input, outputs=predictions_resnet)
       
    return inception_model, vgg_model, resnet_model


def compile_and_train(model, model_name, WIDTH=224, HEIGHT=224, BATCH_SIZE=64):
    
    # Define the optimizer
    sgd = optimizers.SGD(nesterov=True)

    # Compile model
    model.compile(optimizer=sgd, loss='categorical_crossentropy', 
            metrics=['accuracy', precision_metric, recall_metric, f1_metric])
    
    # Train data augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=50,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Validation set preparation
    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Training set generator
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(HEIGHT, WIDTH),
        shuffle=True,
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    
    # Validation set generator
    validation_generator = validation_datagen.flow_from_directory(
        'dataset/cross_val',
        target_size=(HEIGHT, WIDTH),
        shuffle=True,
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    # Define some hyperparameters
    EPOCHS = 15
    STEPS_PER_EPOCH = train_generator.samples // BATCH_SIZE * 3
    VALIDATION_STEPS = validation_generator.samples // BATCH_SIZE 
    
    # Balance classes
    class_weights = class_weight.compute_class_weight('balanced', 
            np.unique(train_generator.classes), train_generator.classes)

    # Train model
    history = model.fit_generator(
        train_generator,
        epochs=EPOCHS,
        class_weight=class_weights,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_generator,
        validation_steps=VALIDATION_STEPS)
  
    # Save model to disk
    model.save(model_name + '.model')

    # Save plot with accuracy and loss over epochs
    save_plot(history, model_name)

    return model


def ensemble(models, model_input):
    # Average the output layers of all models
    outputs = [model(model_input) for model in models]
    model_output = Average()(outputs)
    
    # Define the ensemble model
    model = Model(inputs=model_input, outputs=model_output, name='ensemble')
    
    return model


def binary_metrics(conf_matrix, change_class, max_class):
    # Extract binary metrics from a multiclass classification

    tp = conf_matrix[0:change_class, 0:change_class].sum()
    fn = conf_matrix[change_class:max_class+1, 0:change_class].sum()
    fp = conf_matrix[0:change_class, change_class:max_class+1,].sum()
    tn = conf_matrix[change_class:max_class+1, change_class:max_class+1].sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    f1score = (2 * precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1score


def evaluate_model(model, HEIGHT=224, WIDTH=224, BATCH_SIZE=64, to_binary=False):
    # Test set preparation
    test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

    # Test set generator
    test_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(HEIGHT, WIDTH),
        shuffle=True,
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    
    steps_per_epoch = np.math.ceil(test_generator.samples 
            / test_generator.batch_size)
    
    # Predicted labels
    pred = model.predict_generator(test_generator,
            steps=steps_per_epoch, verbose=1)
    y_pred = np.argmax(pred, axis=1)
    
    # True labels
    y_true = test_generator.classes
    
    # Compute accuracy, precision, recall and f1score
    if to_binary:
        cm = confusion_matrix(y_true, y_pred)
        return binary_metrics(cm, 1, 2)


    precision, recall, f1score, _ = precision_recall_fscore_support(y_true, 
            y_pred, average='binary')
    
    return accuracy_score(y_true, y_pred), precision, recall, f1score


if __name__== "__main__":
    
    #prepare_sets()
    
    # Define the input layer shape
    model_input = Input(shape=(224, 224, 3))

    # Prepare the TL models
    inception, vgg, resnet = prepare_models(model_input)
    
    # Compile and re-train all models
    inception = compile_and_train(inception, 'inception')
    vgg = compile_and_train(vgg, 'vgg')
    resnet = compile_and_train(resnet, 'resnet')
   
    # Only needed for load models
    #custom_objects = {'precision_metric': precision_metric, 
    #        'recall_metric': recall_metric, 'f1_metric': f1_metric}
    

    print('InceptionV3 2 classes')
    #inception = load_model('models/tl/inception.model', custom_objects=custom_objects)
    print(evaluate_model(inception))
    
    print('VGG19 2 classes')
    #vgg = load_model('models/tl/vgg.model', custom_objects=custom_objects)
    print(evaluate_model(vgg))
    
    print('ResNet50 2 classes')
    #resnet = load_model('models/tl/resnet.model', custom_objects=custom_objects)
    print(evaluate_model(resnet))
    
    print('Ensemble 3 models')
    ensemble_model = ensemble([inception, vgg, resnet])
    print(evaluate_model(ensemble_model, model_input))
    
    print('Ensemble InceptionV3 + VGG19')
    ensemble_model = ensemble([inception, vgg])
    print(evaluate_model(ensemble_model, model_input))

    print('Ensemble InceptionV3 + ResNet50')
    ensemble_model = ensemble([inception, resnet])
    print(evaluate_model(ensemble_model, model_input))
    
    print('Ensemble VGG19 + ResNet50')
    ensemble_model = ensemble([vgg, resnet])
    print(evaluate_model(ensemble_model, model_input))
    
    # remove test and train paths
    #shutil.rmtree('train') 
    #shutil.rmtree('cross_val') 
    #shutil.rmtree('test') 
