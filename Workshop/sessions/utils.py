import matplotlib.pyplot as plt
import numpy as np
import itertools
import keras.backend as K

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_samples(X_train,N=5):

    '''
    Plots N**2 randomly selected images from training data in a NxN grid
    '''
    import random
    ps = random.sample(range(0,X_train.shape[0]), N**2)

    f, axarr = plt.subplots(N, N)

    p = 0
    for i in range(N):
        for j in range(N):
            if len(X_train.shape) == 3:
                axarr[i,j].imshow(X_train[ps[p]],cmap='gray')
            else:
                im = X_train[ps[p]]
                axarr[i,j].imshow(im)
            axarr[i,j].axis('off')
            p+=1


def plot_curves(history,nb_epoch):

    """
    Plots accuracy and loss curves given model history and number of epochs
    """

    fig, ax1 = plt.subplots()
    t = np.arange(0, nb_epoch, 1)

    ax1.plot(t,history.history['acc'],'b-')
    ax1.plot(t,history.history['val_acc'],'b*')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('acc', color='b')
    ax1.tick_params('y', colors='b')
    plt.legend(['train_acc', 'test_acc'], loc='lower left')
    ax2 = ax1.twinx()
    ax2.plot(t, history.history['loss'], 'r-')
    ax2.plot(t, history.history['val_loss'], 'r*')
    ax2.set_ylabel('loss', color='r')
    ax2.tick_params('y', colors='r')
    plt.legend(['train_loss','test_loss'], loc='upper left')


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
