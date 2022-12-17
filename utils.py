import matplotlib.pyplot as plt 

def plot_accuracy_loss(history):
    fig,ax = plt.subplots(1,2,figsize=(10,8))
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title("model accuracy")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend(['train', 'test'], loc='upper left')

    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('model loss')
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend(['train', 'test'], loc='upper left')
    
    fig.show()

