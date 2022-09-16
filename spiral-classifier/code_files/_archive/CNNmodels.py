# import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# build and train a new model
def trainModel(trainAug, trainAugLabel, valAug, valAugLabel, modelName=None):
    reg = tf.keras.regularizers.l2(0.001)               # include a regularizer to help prevent overfitting
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)  # use the Adam optimizer and set an effective learning rate 

    # build a model
    model = Sequential([
        Conv2D(32, (3,3), padding='same', strides=(1,1), dilation_rate = 1, activation='relu', kernel_regularizer=reg, input_shape=(128,128,1)),
        MaxPool2D((3,3), strides=(1,1)),
        Conv2D(32, (3,3), padding='same', strides=(2,2), dilation_rate = 1, activation='relu', kernel_regularizer=reg),
        MaxPool2D((5,5), strides=(1,1)),
        Conv2D(64, (5,5), padding='same', strides=(1,1), dilation_rate = 2, activation='relu', kernel_regularizer=reg),
        MaxPool2D((5,5), strides=(2,2)),
        Conv2D(128, (7,7), padding='same', strides=(2,2), dilation_rate = 1, activation='relu', kernel_regularizer=reg),
        MaxPool2D((5,5), strides=(1,1)),
        Flatten(),

        ## include some fully connected layers
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(2,activation='softmax') # softmax used for classification, sigmoid better for regression
    ])

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    ## Train the model
    trained_model = model.fit(np.array(trainAug), trainAugLabel, batch_size=128, epochs=35, validation_data=(np.array(valAug), valAugLabel))

    if modelName!=None:
        # save model
        savenameh5 = modelName + '.h5'
        savenametf = modelName + '.tf'
        model.save(savenameh5)
        model.save(savenametf)

    # -------------------------------------
    # plot and save the results
    # Accuracy and Validation Accuracy
    accuracy1 = trained_model.history['accuracy']
    val_acc1 = trained_model.history['val_accuracy']
    epochs = range(len(accuracy1))

    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(18,8)
    ax[0].plot(epochs, accuracy1, 'b', label='Training Accuracy')
    ax[0].plot(epochs, val_acc1, 'r', label='Validation Accuracy')
    ax[0].title.set_text('Accuracy Graph')
    ax[0].legend()
    ax[0].grid()

    # Loss and Validation Loss
    loss1 = trained_model.history['loss']
    val_loss1 = trained_model.history['val_loss']

    ax[1].plot(epochs, loss1, 'b', label='Training Loss')
    ax[1].plot(epochs, val_loss1, 'r', label='Validation Loss')
    ax[1].title.set_text('Loss Graph')
    ax[1].legend()
    ax[1].grid()

    savefigName = modelName + 'accuracy_loss_graph.png'
    ax.savefig(savefigName)


# import a pre-existing model to run on test data (takes h5 model as input)
def importModel(filename, testAug, testAugLabel):
    modelPath = 'savedModels/saved_h5_models/' + filename
    testModel = tf.keras.models.load_model(modelPath)

    loss, acc = testModel.evaluate(np.array(testAug), testAugLabel, verbose=2)
    print("Loss: ", loss, "| Accuracy: ", acc)

    return testModel
    

# use resnet50 model