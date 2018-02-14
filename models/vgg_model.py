from keras.applications.vgg16 import VGG16
import numpy as np
from six.moves import range
from keras.layers import Input, Dropout, Flatten
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.optimizers import Adadelta
from keras import callbacks
import h5py
# import os


# TENSORFLOW: USE CPU INSTEAD OF GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def subtract_mean(img):
    for i in range(img.shape[0]):
        img[i] -= img[i].mean()
    return img


def VGG_model(weights = None, from_scratch = False):

    if from_scratch == False:
        weights = 'imagenet'


    num_classes = 11
    VGG16_Model = VGG16(weights=weights, include_top=False)
    VGG16_Model.summary()

    # Uncomment to unfreeze all layers except the top (lowers performance with current implementation)
    # for layer in VGG16_Model.layers[:]:
    #     layer.trainable = False

    input = Input(shape=(48,48,3), name = 'image_input')

    #Use the generated model
    output_VGG16_Model = VGG16_Model(input)

    #Add the fully-connected layers
    x = Flatten(name='flatten')(output_VGG16_Model)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.25, name="dropout1")(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.25, name="dropout2")(x)
    x = Dense(512, activation='relu', name='fc3')(x)
    x = Dropout(0.25, name="dropout3")(x)

    #Create 5 dense layers with softmax activations to classify 5 possible digit locations
    digit1 = Dense(units = num_classes, activation="softmax", name="output1")(x)
    digit2 = Dense(units = num_classes, activation="softmax", name="output2")(x)
    digit3 = Dense(units = num_classes, activation="softmax", name="output3")(x)
    digit4 = Dense(units = num_classes, activation="softmax", name="output4")(x)
    digit5 = Dense(units = num_classes, activation="softmax", name="output5")(x)

    out_put = [digit1, digit2, digit3, digit4, digit5]

    model = Model(input=input, output=out_put)

    if weights_file:
        model.load_weights(weights_file)

    model.summary()

    print('VGG Model initialized...')

    return model


if __name__ == "__main__":


    # Include negative samples so CNN can perform detection
    # All samples are size 48 x 48 x 1, labels are 0-9 with 10 representing no digit
    weights_file = "saved/transfer_weights.h5"
    training_imgs = 'data/SVHN_BGR_48.h5'
    negative_samples = 'data/SVHN_neg_BGR_48.h5'


    # Load Data from H5 files
    h5f = h5py.File(training_imgs, 'r')
    X_train = h5f['train_dataset'][:]
    y_train = h5f['train_labels'][:]
    X_val = h5f['valid_dataset'][:]
    y_val = h5f['valid_labels'][:]
    X_test = h5f['test_dataset'][:]
    y_test = h5f['test_labels'][:]
    h5f.close()

    # Load Dataset of negative examples
    h5f2 = h5py.File(negative_samples, 'r')
    X_train_neg = h5f2['train_dataset_neg'][:]
    y_train_neg = h5f2['train_labels_neg'][:]
    h5f2.close()

    # Add Negative Examples to training data
    X_train = np.vstack((X_train, X_train_neg))
    y_train = np.vstack((y_train, y_train_neg))

    # Shuffle Examples (Optional)
    p = np.random.permutation(y_train.shape[0])
    X_train = X_train[p]
    y_train = y_train[p]

    # Subtract mean from training data
    X_train2 = subtract_mean(X_train.astype('float32'))
    X_val2 = subtract_mean(X_val.astype('float32'))

    # Rearrange Y values to match last layer of network
    y_temp = np.copy(y_train).transpose()
    y0 = y_temp[0]
    y1 = y_temp[1]
    y2 = y_temp[2]
    y3 = y_temp[3]
    y4 = y_temp[4]
    y_train2 = [y0, y1, y2, y3, y4]

    # Repeat for Validation values
    y_temp = np.copy(y_val).transpose()
    y0 = y_temp[0]
    y1 = y_temp[1]
    y2 = y_temp[2]
    y3 = y_temp[3]
    y4 = y_temp[4]
    y_val2 = [y0, y1, y2, y3, y4]

    model = VGG_model(weights = weights_file, from_scratch=False)

    optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    #Early stopping critera, model stops training when validation error becomes larger

    checkpoint = ModelCheckpoint(filepath=weights_file, monitor='val_acc', verbose=1, save_best_only=True)

    stopping = EarlyStopping(monitor='val_loss', patience=3, mode='auto')

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, \
                                  patience=2, min_lr=0.001)

    tbCallBack = callbacks.TensorBoard(log_dir='checkpoints/VGG_Transfer',\
                                       histogram_freq=0, write_graph=True, write_images=True)

    callbacks = [checkpoint, stopping, tbCallBack, reduce_lr]

    training_stats = model.fit(x=X_train2, y=y_train2, validation_data=(X_val2, y_val2), epochs=10, \
                               batch_size=128, verbose=1, callbacks=callbacks)

    #Save model and weights
    model.save_weights(weights_file)
    model.save('VGG_Transfer.h5')







