import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import SGD;
import OBFGS;
import NOBFGS;
from sklearn.metrics import accuracy_score;
from sklearn.utils import shuffle
#%%
# ---------------------------------------------------------------------------------------- #
                        #  Support Vector Machines
# ---------------------------------------------------------------------------------------- #

def loss(W,x,y):
    arg = 1-y*np.inner(W,x)
    gamma = 0.1;
    return np.maximum(0,arg) + 0.1*gamma*np.linalg.norm(W)**2;

def loss_gradient(W,x,y):
    gamma = 0.1;
    arg = 1-y*np.inner(W,x);
    if(arg > 0):
        return -y*x + gamma*W;
    return gamma*W;

def predict(W,X):
    return np.sign(np.dot(X,W));

#%%
def main(dataType):
    print(f"Solving the {dataType} problem")
    # ---------------------------------------------------------------------------------------- #
                            # Data Selection Section
    # ---------------------------------------------------------------------------------------- #
    if(dataType == "Synthetic"):
        X = np.loadtxt("features.txt");
        y = np.loadtxt("labels.txt");
    else:
        data = pd.read_csv("heart.csv")
        X = np.array(data.iloc[:,0:13]);
        y = np.array(data['target'])


    X, y = shuffle(X, y, random_state=0)

    # Change the ground truths from [0,1] to [-1,1]
    y = np.where(y==1,1,-1)

    # Add the ones to the end for the bias
    X = np.concatenate((X, np.ones((len(X),1))), axis=1);

    # Split the Sample space into training and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)


    # Train the classifiers (Minimize the loss function) with varying batch sizes
    batch_sizes = [1,2,4,6,8,16,32,64,128];
    sgd_accuracies = [];
    obfgs_accuracies = [];
    nobfgs_accuracies = [];
    for size in batch_sizes:
        #print("Solving with Stochastic Gradient Descent")
        W = SGD.Minimize(loss,loss_gradient,0.001,50,size,(X_train,y_train));
        y_pred = predict(W,X_test);
        sgd_accuracies.append( accuracy_score(y_test, y_pred) );
        #print("Solving with Stochastic BFGS")
        W = OBFGS.Minimize(loss,loss_gradient,50,size,(X_train,y_train));
        y_pred = predict(W,X_test);
        obfgs_accuracies.append( accuracy_score(y_test, y_pred) );
        #print("Solving with Nesterov Stochastic BFGS")
        W = NOBFGS.Minimize(loss,loss_gradient,50,size,(X_train,y_train));
        y_pred = predict(W,X_test);
        nobfgs_accuracies.append( accuracy_score(y_test, y_pred) );

    # Plot the graphs
    plt.figure(figsize=(10,5))
    plt.title("Accuracy per batch size")
    plt.xlabel("Batch size")
    plt.ylabel("Accuracy")
    plt.plot(batch_sizes,sgd_accuracies,label="SGD")
    plt.plot(batch_sizes,obfgs_accuracies,label="sBFGS")
    plt.plot(batch_sizes,nobfgs_accuracies,label="nsBFGS")
    plt.legend(loc=("lower right"))
    plt.show()

#%%
# ----------------------------------------------------------------------------------------------- #
                        # Optimization 2020 Project
            # Accelerated Stochastic BFGS vs Stochastic BFGS vs SGD
            # Authors : Yuval Lavie, Asaf Ahi Moredchai
# Please comment one of the main lines below if you wish to only view a specific data type
# ----------------------------------------------------------------------------------------------- #
main("Synthetic");
main("Heart rate");