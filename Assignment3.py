import numpy as np
import sklearn
from sklearn.ensemble import BaggingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import datasets

#Filtering warnings so output looks more clear
import warnings
warnings.filterwarnings("ignore")

#AHMET ORAL - 180709008
print("ASSIGNMENT 3")

#seed for random state
SEED = 42

#2.1.1 Different training sets
def task2_1_1():
    print("\nTASK 2.1.1:")

    #Loading digits dataset
    X, y = datasets.load_digits(return_X_y=True)

    #Spliting data with 30/70 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

    #Creating instance of MLP with 4 hidden layers and 1000 iterations
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8, 4, 2), max_iter=1000, random_state=SEED)

    #Applying BaggingClassifier(with replacement)
    bcf = BaggingClassifier(mlp, n_estimators=8, random_state=SEED)
    bcf.fit(X_train, y_train)

    #Predicting and getting accuracy score of bagging algorithm
    y_pred= bcf.predict(X_test)
    bagging_error= accuracy_score(y_test,y_pred)

    #For loops that runs on bcf.estimators_to print correctly classified instances of learners
    bcfEstimators = bcf.estimators_
    counter = 0
    for i in bcfEstimators:

        #Predicting
        pred = i.predict(X_test)

        #Taking accuracy score
        error = accuracy_score(y_test, pred)

        #Getting the number of correctly classified intances
        number = int(540 * error)
        print(number, "out of", len(X_test), "instances are correctly classified by learner #", counter)


    print("------------------------------------------------")

    #For bagging
    print(int(540*bagging_error),"out of",len(X_test),"instances are correctly classified by bagging" )

#2.1.2 Boosting
def task2_1_2():
    print("\nTASK 2.1.2:\nShowing the Figure")

    #Loading moons dataset with more than 100 tuples
    #I added Gaussian Noise by adding 'noise = 0.2' parameter
    X, y = datasets.make_moons(n_samples= 140, noise=0.2, random_state=SEED)

    #Spliting data with 30/70 ratio
    X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=SEED)

    #Creating instance of logistig regression algorithm with SGD solver
    sgd = SGDClassifier(loss='log', random_state=0).fit(X_train, y_train)

    #Applying AdaBoost classifier with 4 base classifier to sgd
    acf = AdaBoostClassifier(n_estimators=4, base_estimator=sgd, random_state=SEED).fit(X_train, y_train)

    #I create 2 list that contains coefValues and interceptValues.By doing this I can use the values of it.
    coefValues = []
    interceptValues = []
    for i, j in zip(acf.estimators_, acf.estimator_weights_):
        coefValues.append(i.coef_)
        interceptValues.append(i.intercept_)

    #Function to create the Hypothesis line
    def hyphothesis(x):
        alfa = -coefValues[x][0][0]/coefValues[x][0][1]
        line = alfa*np.linspace(-1, 2) - (interceptValues[x][0]) / coefValues[x][0][1]
        return line

    #Defining figure
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))

    #--------------------------------------------------------------------------------------------------
    #Graph of learner 1
    ax[0].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='blue', marker='^', alpha=0.7)
    ax[0].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='red', marker='o', alpha=0.7)

    #Hypothesis
    ax[0].plot((np.linspace(-1, 3)), hyphothesis(0), "--")
    #--------------------------------------------------------------------------------------------------

    # Graph of learner 2
    ax[1].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='blue', marker='^', alpha=0.7)
    ax[1].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='red', marker='o', alpha=0.7)

    # Hypothesis
    ax[1].plot((np.linspace(-1, 3)), hyphothesis(1), "--")
    #--------------------------------------------------------------------------------------------------

    # Graph of learner 3
    ax[2].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='blue', marker='^', alpha=0.7)
    ax[2].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='red', marker='o', alpha=0.7)

    # Hypothesis
    ax[2].plot((np.linspace(-1, 3)), hyphothesis(2), "--")
    #--------------------------------------------------------------------------------------------------

    # Graph of learner 3
    ax[3].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='blue', marker='^', alpha=0.7)
    ax[3].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='red', marker='o', alpha=0.7)

    # Hypothesis
    ax[3].plot((np.linspace(-1, 3)), hyphothesis(3), "--")
    #--------------------------------------------------------------------------------------------------

    #Setting titles
    ax[0].set_title('Learner #1')
    ax[1].set_title('Learner #2')
    ax[2].set_title('Learner #3')
    ax[3].set_title('Learner #4')

    plt.tight_layout()
    #Saving and closing pdf
    fig.savefig('BaseLearnerVisualization.pdf', format='pdf')
    plt.show()
    plt.close()


#2.2 Different learning algorithm
def task2_2():
    print("\nTASK 2.2:\nShowing the Figure")

    #Loading the dataset
    X, y = datasets.load_breast_cancer(return_X_y=True)

    #Here I split data with 20/80 ratio because in introduction you said we use an approach that uses 20/80 ratio
    #And you didn't specifically ask for us to use 30/70 ratio.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    #Using with 5 fold
    cv = KFold(n_splits=5, random_state=SEED)

    #classifier1 with LogisticRegression
    classifier1 = LogisticRegression(multi_class='multinomial', random_state=SEED)
    #Fitting X_train,y_train
    classifier1.fit(X_train,y_train)
    #Getting error value with cross val score using cv and printing
    score1 = cross_val_score(classifier1, X_test, y_test, cv=cv, scoring='accuracy')
    print('Accuracy obtained by learner #2 is:', np.mean(score1))

    # classifier2 with SGDClassifier
    classifier2 = SGDClassifier(random_state=SEED)
    # Fitting X_train,y_train
    classifier2.fit(X_train,y_train)
    #Getting error value with cross val score using cv and printing
    score2 = cross_val_score(classifier2, X_test, y_test, cv=cv, scoring='accuracy')
    print('Accuracy obtained by learner #2 is:', np.mean(score2))

    #classifier3 with SVC
    classifier3 = SVC(random_state=SEED)
    # Fitting X_train,y_train
    classifier3.fit(X_train,y_train)
    #Getting error value with cross val score using cv and printing
    score3 = cross_val_score(classifier3, X_test, y_test, cv=cv, scoring='accuracy')
    print('Accuracy obtained by learner #3 is:', np.mean(score3))


    print("---------------------------------------------")

    #VotingClassifier learner with 3 classifiers
    Vclf = VotingClassifier(estimators=[('lr',classifier1), ('sgd',classifier2), ('svc',classifier3)], voting='hard')
    Vclf.fit(X_train,y_train)

    #Getting the cross_val_score score
    crosValScore = cross_val_score(classifier1, X_test, y_test, scoring='accuracy', cv=cv)

    # Printing the score of vot_error
    print('Accuracy obtained by ensemble learner is:', crosValScore.mean())



#2.3 Different parameter setting
def task2_3():
    print("\nTASK 2.3:")

    #Loading the dataset
    X, y = datasets.load_breast_cancer(return_X_y=True)

    #Spliting data with 30/70 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

    #Counter
    counter = 0

    #Function to return neurons values with respect to hidden layer size
    def h_values(h):
        list = []
        for i in range(h):
            i = i + 1
            list.append(2 ** i)
        list.reverse()
        return list

    #For loop
    for j in range(10):
        #MLPClassifier with 10 hidden layer size with asked parameter.
        clf = MLPClassifier(random_state=SEED, hidden_layer_sizes=(h_values(j)))

        #Fitting X_train, y_train
        clf.fit(X_train, y_train)

        #Predicting
        y_pred = clf.predict(X_test)

        #Getting the accuracy score
        score = accuracy_score(y_test, y_pred)

        #Printing score values and updating counter
        counter += 1
        print("Parameter setting: l#", counter, "Accuracy:", score)

    #VotingClassifier with MlpClassifier
    Vclf = VotingClassifier(estimators=[('mlp', clf)], voting='hard')

    #Fitting X_train, y_train
    Vclf.fit(X_train, y_train)

    #Getting the prediction
    y_pred = Vclf.predict(X_test)


    print("-----------------------------------------")

    #Printing the score of ensamble learning
    print("Ensemble Learning Accuracy:", accuracy_score(y_test, y_pred))


#3 k-Nearest Neighbors Classifier
def task3():
    print("\nTASK 3:\nShowing the Figure")

    #Loading moons datset with more than 100 tuples and adding GaussianNoise with 'noise = 0.3' parameter
    X, y = datasets.make_moons(n_samples= 150, noise=0.3, random_state=SEED)

    #Spliting data such that only four tuples are used for testing while remaining tuples are used for training.
    X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=4, random_state=SEED)

    #Applying k-NN classifier for each testing tuples with k=5
    KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

    #Function to return nearest k neighbors of that sample
    def nearest_k_neighbors(train, test, k=5):
        nbors = [(a, np.sum(np.abs(a - test))) for a in train]
        nbors = sorted(nbors, key=lambda a: a[1])
        return np.array([a for a, _ in nbors[:k]])

    #Defining the figure
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))

    #-------------------------------------------------------------------
    #Figure1:

    #Getting nearest 5 neighbors of that sample
    nearest_neighbors = nearest_k_neighbors(X_train, X_test[0], k=5)

    ax[0].scatter(X_test[0][0], X_test[0][1], color='black', marker='+', s=700)
    ax[0].plot(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 'bx')
    ax[0].plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 'ro')

    #Showing nearest 5 neighbors of that sample
    ax[0].scatter(nearest_neighbors[:, 0], nearest_neighbors[:, 1], color="green", s=70)
    #-------------------------------------------------------------------

    # Figure2:

    #Getting nearest 5 neighbors of that sample
    nearest_neighbors = nearest_k_neighbors(X_train, X_test[1], k=5)

    ax[1].scatter(X_test[1][0], X_test[1][1], color='black', marker='+', s=700)
    ax[1].plot(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 'bx')
    ax[1].plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 'ro')

    #Showing nearest 5 neighbors of that sample
    ax[1].scatter(nearest_neighbors[:, 0], nearest_neighbors[:, 1], color="green", s=70)

    #Getting nearest 5 neighbors of that sample
    nearest_neighbors = nearest_k_neighbors(X_train, X_test[2], k=5)
    #-------------------------------------------------------------------

    # Figure3:
    ax[2].scatter(X_test[2][0], X_test[2][1], color='black', marker='+', s=700)
    ax[2].plot(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 'bx')
    ax[2].plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 'ro')

    #Showing nearest 5 neighbors of that sample
    ax[2].scatter(nearest_neighbors[:, 0], nearest_neighbors[:, 1], color="green", s=70)

    #Getting nearest 5 neighbors of that sample
    nearest_neighbors = nearest_k_neighbors(X_train, X_test[3], k=5)
    #-------------------------------------------------------------------

    #Figure4:
    ax[3].scatter(X_test[3][0], X_test[3][1], color='black', marker='+', s=700)
    ax[3].plot(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 'bx')
    ax[3].plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 'ro')

    #Showing nearest 5 neighbors of that sample
    ax[3].scatter(nearest_neighbors[:, 0], nearest_neighbors[:, 1], color="green", s=70)
    #-------------------------------------------------------------------

    #Setting the titles
    ax[0].set_title('Test object#1 \npred. class: 0')
    ax[1].set_title('Test object#2 \npred. class: 0')
    ax[2].set_title('Test object#3 \npred. class: 0')
    ax[3].set_title('Test object#4 \npred. class: 1')

    #Showing saving and closing figure
    plt.tight_layout()
    fig.savefig('KNN.pdf', format='pdf')
    plt.show()
    plt.close()


#Calling Functions of each tasks

task2_1_1()
task2_1_2()
task2_2()
task2_3()
task3()