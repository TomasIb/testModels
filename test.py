import csv
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix

DATASET_ROUTE ='./test2/test-modelo2.csv'
MODEL ='./test2/best_modelo2.sav'



def generateConfusionMatrix(cm):
    plt.figure(figsize=(6.25,6.25))
    sn.heatmap(cm, cmap="Blues",annot=True,cbar=False,fmt='g')
    plt.title("Test")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    
    plt.savefig('./confusionMatrix.png', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1) 


def run():
    sniffer = csv.Sniffer()
    sample_bytes = 32
    dialect = sniffer.sniff(open(DATASET_ROUTE).read(sample_bytes))           

    test_dataset = pd.read_csv(DATASET_ROUTE,sep=dialect.delimiter)      
    test_dataset=test_dataset.iloc[:,1:]
    X_test = np.array(test_dataset.iloc[:,:-1])
    y_test = np.array(test_dataset.iloc[:,-1]) 

    loaded_model = pickle.load(open(MODEL, 'rb'))

    y_pred = loaded_model.predict(X_test)
    y_prob = loaded_model.predict_proba(X_test)


    test_dataset['model'] = y_pred
    test_dataset['model'+' PROB'] = y_prob[:,1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cm = np.array([[tp,fp],[fn,tn]]) 
    generateConfusionMatrix(cm)         

    test_dataset.to_csv('./output.csv', index = None, header=True) 


if __name__ == '__main__':   
    run()