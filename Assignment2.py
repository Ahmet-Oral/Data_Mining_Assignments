import time
import sklearn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Perceptron, Ridge
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#Ahmet Oral - 180709008

#I had few unimportant warning so I disabled them.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


###################################################
############2.1 Classification task################
###################################################

#Creating a fixed seed value for all seeds in this program.
SEED = 42

#Tuple and Dimension values for task a,b,c,d.
a = [10000,100]
b = [10000,1000]
c = [100000,100]
d = [250000,100]
print("ASSIGNMENT 2.1")
print("Single-layer perceptron with 100 iterations:")

#For loop to calculate running time and error for 100 iterations.
for i,j in [a,b,c,d]:

    #Creating a dataset with requested n_sample and n_feature's
    n_sample = i
    n_feature = j
    X, Y= sklearn.datasets.make_classification(n_samples=n_sample, n_features=n_feature,random_state=SEED,shuffle=False )

    # Starting Counter to calculate total running time
    start_time = time.time()

    #train_test_split with test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=SEED)

    # single layer perceptron with 100 iterations
    p = Perceptron(random_state=SEED,max_iter=100)
    p.fit(X_train, y_train)
    y_predicted = p.predict(X_test)

    #Printing the running time and error.
    print(i,"Tuples and",j,"Dimensions -->"," Running Time: %s seconds" % (time.time() - start_time),"- Error is: ",mean_squared_error(y_test, y_predicted))
#End of Loop

print("Single-layer perceptron with 500 iterations:")

#Tuple and Dimension values for task e,f,g,h.
e = [10000,100]
f = [10000,1000]
g = [100000,100]
h = [250000,100]

#For loop to calculate running time and error for 500 iterations.
for i,j in [a,b,c,d]:

    # Creating a dataset with requested n_sample and n_feature's
    n_sample = i
    n_feature = j
    X, Y= sklearn.datasets.make_classification(n_samples=n_sample, n_features=n_feature,random_state=SEED,shuffle=False )

    # Starting Counter to calculate total running time
    start_time = time.time()

    #train_test_split with test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=SEED)

    #single layer perceptron with 500 iterations
    p = Perceptron(random_state=SEED,max_iter=500)
    p.fit(X_train, y_train)
    y_predicted = p.predict(X_test)

    #Printing the running time and error.
    print(i,"Tuples and",j,"Dimensions -->"," Running Time: %s seconds" % (time.time() - start_time),"- Error is: ",mean_squared_error(y_test, y_predicted))
#End of loop

print("\n END OF ASSIGNMENT 2.1")
print("Showing Results of 2.2:")


###################################################
#####2.2  Visualization of decision boundary#######
###################################################

#n_ sample and n_feature values are at least 500 tuples and 3 dimensions.
n_sample = 600
n_feature = 9

#I created data with n_informative = 2 so two features can be informative to the ground truth vector.
X, y= sklearn.datasets.make_classification(n_samples=n_sample, n_features=n_feature,n_informative=2,random_state=SEED)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

#I used MLPClassifier with no hidden_layer_sizes instead of perceptron because when I used perceptron programs crashes I don't completely understand why.
mlp = MLPClassifier(hidden_layer_sizes=(), random_state=SEED)
clf = mlp.fit(X_test, y_test)


z = lambda x, y: (-clf.intercepts_[0] - clf.coefs_[0][0] * x - clf.coefs_[0][1] * y) / clf.coefs_[0][2]

#defining linspace and creating meshgrid with it.
tmp = np.linspace(-5, 5, 30)
x, y = np.meshgrid(tmp, tmp)

#creating figure with X_test and y_test values.
fig = plt.figure( figsize=(5,5) )
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[y_test==0,0]+2, X_test[y_test==0,1]-2, X_test[y_test==0,2]-5, c='b', marker='^', s=25)
ax.scatter(X_test[y_test==1,0]-2, X_test[y_test==1,1]+2, X_test[y_test==1,2]+5, c='r', marker='o', s=25)

#displaying figure
ax.plot_surface(x, y, z(x, y))
ax.view_init(30, 60)
plt.show()

print("\n END OF ASSIGNMENT 2.2")
print("Showing Results of 3.1:")


###################################################
#3.1 Error convergence with multi-layer perceptron#
###################################################

#Loading digits dataset
X, y = datasets.load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=SEED)

#MLPClassifier, 1 hidden layer size with 50 neurons
clf = MLPClassifier(max_iter=100, hidden_layer_sizes=(50,), random_state=SEED)
clf.fit(X_train, y_train)

#x,y label names and title
plt.ylabel('Error')
plt.xlabel('iterations')
plt.title("Convergence of error with MLP")

#displaying convergence plot for error values.
plt.plot(clf.loss_curve_)
plt.show()

print("\n END OF ASSIGNMENT 3.1")
print(" Showing Results of 3.2:")



#########################################################################
#3.2 Effects of multi-layet perceptron structure on train & test scores##
#########################################################################

#Takin H value from user
#h = int(input("\nPlease enter the 'H' value: "))
h=10 #I put constant 10 instead of taking ipnut.

#Creating 2 empyt list to store train_scores and test_scores.
train_scores = []
test_scores = []

#number of hidden layers and neurons inside them is determined by 'h' value as you defined in assignment paper.
#for example if 'h' is 3 then hidden layer size will be 3 and neurons in them goes as 2**3,2**2,2**1.

#To make this possible I created this h_values function.
#Function starts by creating an empty list,
#...then, starting by 1 it takes the square of numbers and appends them to list until it reaches h.
#So for example if h = 4, list is [2,4,8,16].
#Last step is to reverse the list, so hidden layers goes in the same order you asked.
def h_values(h):
    list=[]
    for i in range(h):
        i=i+1
        list.append(2**i)
    list.reverse()
    return list

#Loading digits dataset
X, y = datasets.load_digits(return_X_y=True)

#I created a for loop to take the scores of all hidden_layer_sizes.
#loop runs h times and appends the train and test scores to the list I created before.
for i in range(h):
    # I created a variable named hidden_layers.I put the hidden layers with neurons in this variable like: (16,8,4,2)
    hidden_layers = h_values(i+1)    #I call the h_values function with i+1 so it does'n start from 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=SEED)

    #Calling MLPClassifier with "hidden_layer_sizes" parameter "hidden_layers"
    clf = MLPClassifier(hidden_layer_sizes=(hidden_layers), random_state=SEED)
    clf.fit(X_train, y_train)

    #Putting train and test scores in a list so we can use this values to create a graph
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

    #printing hidden layer with train and test scores
    print("Hidden layers: ",hidden_layers,"Train scores: ", train_scores[i],"--Test scores: ",test_scores[i])
#End of for loop

#This function is not too important
#I wrote this to create a list of H numbers to use it in graph
#if H is 5 layersize list is [1,2,3,4,5]
def hidden_layer_size(h):
    layersize = []
    for i in range(h):
        i = i+1
        layersize.append(i)
    return layersize #I use the layer size for the x axis values in the graph

#defining the graph
plt.ylabel('Accuracy Score')
plt.xlabel('Hidden Layer Size')
plt.title("Train & test scores as a function of hidden layer size.")
plt.plot(hidden_layer_size(h) , train_scores, label = "Train") #train scores
plt.plot(hidden_layer_size(h) , test_scores, label = "Test") #test scores
plt.show()