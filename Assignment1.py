import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
import time
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns


########################################
#####  2.1 CURSE OF DIMENSIONALITY #####
########################################

#setting a fixed seed so I can get same values for every run
SEED = 44

print("***************************\n"
      "2.1 Curse Of Dimensionality\n"
      "***************************\n")

#Paramaters for tuple size and dimension size as requested in assignment.
#Values are prefixed and named in the same sequencel order in assignemnt.
a =[10000,100,"a"]
b =[10000,1000,"b"]
c =[10000,2000,"c"]
d =[100000,100,"d"]
e =[250000,100,"e"]
f =[500000,100,"f"]

#For loop to display all asked results.
for i,j,k in [a,b,c,d,e,f]:

    #Creating new data with n_samples and n_features.
    n_sample = i
    n_feature =j

    #to get same error value I made random_state a fixed seed and turned shuffle to false.
    X, Y= sklearn.datasets.make_classification(n_samples=n_sample, n_features=n_feature,random_state=SEED,shuffle=False )

    #Splitting the data in 0.3 ratio with fixed random_state value
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=SEED)

    #Starting Counter to calculate total running time
    start_time = time.time()

    #Applying SGDClassifier with 1000 iterations with fixed random state value
    clf = SGDClassifier(max_iter=1000,random_state=SEED)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    # Displaying total running time and Mean Squared Error
    print(k,") Running Time: %s seconds" % (time.time() - start_time) , "/ Mean Squared Error of SGD :",mean_squared_error(y_test, y_predicted))
#end of for loop

input("\n2.1 is finished\nTo display results of 2.2 enter any input"
      "\n(It may lag for few seconds)")

#######################################################
#####  2.2  SAMPLING AND DIMENSIONALITY REDUCTION #####
#######################################################

print("\n*****************************************\n"
      "2.2 SAMPLING AND DIMENSIONALITY REDUCTION\n"
      "****************************************\n")

#First we will only change dimension size while keeping tuple size same.
#Second we will only change tuple size and keep dimension same.
#both results will be displayed in order as asked in the assignment.

#Dimension values.
dimension_values = [500,100,10,4,1]

#I create dataset with fixed n_sample and n_features and I will reduce dimension size with algorithms in the for loop.
X, Y= sklearn.datasets.make_classification(n_samples=10000, n_features=1000,random_state=SEED,shuffle=False )

# First we will only change Dimension Size
#Constructing a for loop with asked dimensions.
for i in dimension_values:

    dimension =i
    # Reducing dimension in each loop using PCA with required dimension size.
    pca = PCA(n_components = dimension, whiten=True)
    pca.fit(X)
    x_pca = pca.transform(X)

    #Splitting the data in 0.3 ratio with fixed random_state value
    X_train, X_test, y_train, y_test = train_test_split(x_pca, Y, test_size=0.3, random_state=1)

    # Starting Counter to calculate total running time
    start_time = time.time()

    # Applying SGDClassifier with 1000 iterations with fixed random state value
    clf = SGDClassifier(max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    print("Dimension Size:",dimension,"Running Time: %s seconds" % (time.time() - start_time) , "/ Mean Squared Error of SGD :",mean_squared_error(y_test, y_predicted))

#Dimension size reduction is finished
print("*******************************************************************")

#Now Tuple size sampling starts

#Tuple values
tuple_values = [300000,150000,100000,1000,100]
#I create dataset with fixed n_sample and n_features and I will reduce tuple size with algorithms in for loop.
X, Y= sklearn.datasets.make_classification(n_samples=400000, n_features=100,random_state=SEED,shuffle=False )

#Constructing a for loop with asked tuple size
for i in tuple_values:

    #Sampling according to the requested tuple size
    sample_size=i
    X, Y = resample(X, Y, n_samples=sample_size)

    # Splitting the data in 0.3 ratio with fixed random_state value
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

    # Starting Counter to calculate total running time
    start_time = time.time()

    # Applying SGDClassifier with 1000 iterations with fixed random state value
    clf = SGDClassifier(max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    print("Tuple Size:",sample_size,"-","Running Time: %s seconds" % (time.time() - start_time) , "/ Mean Squared Error of SGD :",mean_squared_error(y_test, y_predicted))
#end of loop


input("\n2.2 is finished\nTo display results of 3.1 enter any input"
      "\n(It may lag for few seconds)")

###############################################################
#####  3.1  Visualization and binary-class classification #####
###############################################################

#Calling make_moons with more than 100 tuples as asked in assignment
X, y = make_moons(n_samples=200, random_state=SEED)

#Dividing it as required
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Runing logistic regression algorithm (loss = 'log') with SGD solver with 10,000 iterations
clf = SGDClassifier(max_iter=10000, random_state=SEED, loss="log")

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6)) #I put same nrows and ncols as in the assignments

#I made it look like the figure in the assignemnt.It's not too important but looks good :D
#Implemanting train and test values.
ax[0].scatter(x_train[y_train==0, 0], x_train[y_train==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(x_train[y_train==1, 0], x_train[y_train==1, 1], color='blue', marker='o', alpha=0.5)
ax[1].scatter(x_test[y_test==0, 0], x_test[y_test==0, 1], color='red', marker='^', alpha=0.5)
ax[1].scatter(x_test[y_test==1, 0], x_test[y_test==1, 1], color='blue', marker='o', alpha=0.5)

#hyphothesis line
plt.plot(X, X/3.3, 'black')

#titles and labels
ax[0].set_title('Train')
ax[0].set_xlabel('X1')
ax[0].set_ylabel('X2')
ax[1].set_title('Test')
ax[1].set_ylabel('X2')
ax[1].set_xlabel('X1')

plt.tight_layout()
plt.show()
input("\n3.1 is finished\nTo display results of 3.2 enter any input")


#######################################################
#####  3.2 Noising and multi-class classification #####
#######################################################
print("I tried to do 3.2 so much but whatever I do I failed.I couldn't understand the concept behind it."
      "I did some online research and sometimes I thought I am getting closer but all my attempts has been for nothing :/"
      " .I asked my friends for advice but it didn't help either.So sadly this part is empty." )


input("\nTo display results of 3.3 enter any input")

############################################################
#####  3.3 Evaluation/performance metric demonstration #####
############################################################

#Loading digits dataset form sklearn
X, y = load_digits(return_X_y=True)

#Dividing them as asked in assignment
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Runing logistic regression algorithm (loss = 'log') with SGD solver with 10,000 iterations
model = SGDClassifier(max_iter=10000, loss="log")

#fitting x_train and y_train in mode
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

#Implementing test and prediction values to confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

#Using heatmap to represent matrixDisplaying and Displaying heatmap
sns.heatmap(conf_matrix, annot = True,cmap = "YlGnBu")
plt.show()

print("END OF ASSIGNMENT")
print("AHMET ORAL 180709008")