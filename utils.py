import pandas as pd
import numpy as np
from sklearn import metrics 
import matplotlib.pyplot as plt
import seaborn as sns

def save_accuracy_plot(model_history, model_name):
    plt.figure(figsize = (18,8))
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title("Training & Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy','Testing Accuracy'])
    plt.savefig("./results/"+model_name+"_accuracy.png")
    plt.show()

def save_loss_plot(model_history, model_name):
    plt.figure(figsize = (18,8))
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title("Training & Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train loss','Testing loss'])
    plt.savefig("./results/"+model_name+"_loss.png")
    plt.show()

def save_model_history(model_history, model_name):
    save_accuracy_plot(model_history, model_name)
    save_loss_plot(model_history, model_name)


def save_results(y_test, y_pred, model_name="model"):
    with open("./results/"+model_name+"_scores.txt", 'w') as f:
        print(metrics.classification_report(y_test, y_pred), file = f)
        print(metrics.classification_report(y_test, y_pred))
    # Confusion matrix
    #plt.figure(figsize=(10,10))
    cmn=metrics.confusion_matrix(y_test,y_pred)
    #cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #fig, ax = plt.subplots(figsize=(15,15))
    #cmn = np.array(cmn, dtype='int')
    plt.figure(figsize=(15,15))
    sns.heatmap(cmn, annot=True,cmap='Blues', fmt='g')
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("./results/"+model_name+"_confusion.png")
    plt.show(block=False)

    
def load_dataset():
    # load dataset
    train_df = pd.read_csv("data/sign_mnist_train.csv")
    test_df = pd.read_csv("data/sign_mnist_test.csv")
        
    train_arr = train_df.iloc[:, 1:].to_numpy()
    train_label_arr = train_df['label'].to_numpy()
    test_arr = test_df.iloc[:, 1:].to_numpy()
    test_label_arr = test_df['label'].to_numpy()

    X_train=train_arr/255
    X_test=test_arr/255

    y_train = train_label_arr
    y_test  = test_label_arr
    return X_train, y_train, X_test, y_test