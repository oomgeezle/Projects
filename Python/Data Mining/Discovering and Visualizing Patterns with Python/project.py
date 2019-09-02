def DataImportingAndVisualization(data, target):
    
    print(data.shape)
    print(target.shape)
    print(set(target)) # build a collection of unique elements
    set(['setosa', 'versicolor', 'virginica'])

    #Using the plotting capabilities of the pylab library (which is an interface to
    #matplotlib) we can build a bi-dimensional scatter plot which enables us
    #to analyze two dimensions of the dataset plotting the values of a feature
    #against the values of another one
    from pylab import plot, show
    plot(data[target=='setosa',0],data[target=='setosa',2],'bo')
    plot(data[target=='versicolor',0],data[target=='versicolor',2],'ro')
    plot(data[target=='virginica',0],data[target=='virginica',2],'go')
    show()

    #Another common way to look at data is to plot the histogram of the single
    #features. In this case, since the data is divided into three classes, we can
    #compare the distributions of the feature we are examining for each class.
    #With the following code we can plot the distribution of the first feature of
    #our data (sepal length) for each class
    from pylab import figure, subplot, hist, xlim, show
    xmin = min(data[:,0])
    xmax = max(data[:,0])
    figure()
    subplot(411) # distribution of the setosa class (1st, on the top)
    hist(data[target=='setosa',0],color='b',alpha=.7)
    xlim(xmin,xmax)
    subplot(412) # distribution of the versicolor class (2nd)
    hist(data[target=='versicolor',0],color='r',alpha=.7)
    xlim(xmin,xmax)
    subplot(413) # distribution of the virginica class (3rd)
    hist(data[target=='virginica',0],color='g',alpha=.7)
    xlim(xmin,xmax)
    subplot(414) # global histogram (4th, on the bottom)
    hist(data[:,0],color='y',alpha=.7)
    xlim(xmin,xmax)
    show()

def Classification(data, target):
    #The library sklearn contains the implementation of many models for
    #classification and in this section we will see how to use the Gaussian Naive
    #Bayes in order to identify iris flowers as either setosa, versicolor or virginica
    #using the dataset we loaded in the first section. To this end we convert the
    #vector of strings that contain the class into integers
    t = zeros(len(target))
    t[target == 'setosa'] = 1
    t[target == 'versicolor'] = 2
    t[target == 'virginica'] = 3
    #instantiate and train our classifier
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(data,t) # training on the iris dataset
    #test it with one of the sample
    print(classifier.predict(data)[0])
    print(t[0])
    #To this end we split the
    #data into train set and test set, picking samples at random from the original
    #dataset. We will use the first set to train the classifier and the second one to
    #test the classifier. The function train_test_split can do this for us
    from sklearn.model_selection import train_test_split
    train, test, t_train, t_test = train_test_split(data, t, test_size=0.4, random_state=0)
    #The dataset have been split and the size of the test is 40% of the size of the
    #original as specified with the parameter test_size. With this data we can
    #again train the classifier and print its accuracy
    classifier.fit(train,t_train) # train
    print(classifier.score(test,t_test)) # test
    #Another tool to estimate the performance of a classifier is the confusion
    #matrix. In this matrix each column represents the instances in a predicted
    #class, while each row represents the instances in an actual class. Using the
    #module metrics it is pretty easy to compute and print the matrix
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(classifier.predict(test),t_test))
    #A function that gives us a complete report on the performance of the
    #classifier is also available:
    from sklearn.metrics import classification_report
    print(classification_report(classifier.predict(test), t_test, target_names=['setosa', 'versicolor', 'virginica']))
    #To actually
    #evaluate a classifier and compare it with other ones, we have to use a more
    #sophisticated evaluation model like Cross Validation. The idea behind the
    #model is simple: the data is split into train and test sets several consecutive
    #times and the averaged value of the prediction scores obtained with the
    #different sets is the evaluation of the classifier. This time, sklearn provides
    #us a function to run the model:
    from sklearn.model_selection import cross_val_score
    # cross validation with 6 iterations
    scores = cross_val_score(classifier, data, t, cv=6)
    print(scores)
    #We can easily compute the mean accuracy as follows:
    from numpy import mean
    print(mean(scores))

def Clustering(data, target):
    #The library sklearn contains the implementation of many models for
    #classification and in this section we will see how to use the Gaussian Naive
    #Bayes in order to identify iris flowers as either setosa, versicolor or virginica
    #using the dataset we loaded in the first section. To this end we convert the
    #vector of strings that contain the class into integers
    t = zeros(len(target))
    t[target == 'setosa'] = 1
    t[target == 'versicolor'] = 2
    t[target == 'virginica'] = 3
    #One of the most famous clustering tools is the k-means algorithm, which we can
    #run as follows:
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, init='random') # initialization
    kmeans.fit(data) # actual execution
    #Now we can use the model to assign each
    #sample to one of the clusters:
    c = kmeans.predict(data)
    #And we can evaluate the results of clustering, comparing it with the labels
    #that we already have using the completeness and the homogeneity score:
    from sklearn.metrics import completeness_score, homogeneity_score
    print(completeness_score(t,c))
    print(homogeneity_score(t,c))
    #We can also visualize the result of the clustering and compare the
    #assignments with the real labels visually:
    from pylab import figure, subplot, plot, xlim, show
    figure()
    subplot(211) # top figure with the real classes
    plot(data[t==1,0],data[t==1,2],'bo')
    plot(data[t==2,0],data[t==2,2],'ro')
    plot(data[t==3,0],data[t==3,2],'go')
    subplot(212) # bottom figure with classes assigned automatically
    plot(data[c==1,0],data[c==1,2],'bo',alpha=.7)
    plot(data[c==2,0],data[c==2,2],'go',alpha=.7)
    plot(data[c==0,0],data[c==0,2],'mo',alpha=.7)
    show()

def Regression(data, target):
    #In order to apply the linear regression we build a synthetic dataset
    from numpy.random import rand
    x = rand(40,1) # explanatory variable
    y = x*x*x+rand(40,1)/5 # depentend variable
    #Now we can use the LinearRegression model that we found in the module
    #sklear.linear_model. This model calculates the best-fitting line for the
    #observed data by minimizing the sum of the squares of the vertical
    #deviations from each data point to the line. The usage is similar to the other
    #models implemented in sklearn that we have seen before:
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    linreg.fit(x,y)
    #And we can plot this line over the actual data points to evaluate the result:
    from pylab import figure, subplot, plot, xlim, show
    from numpy import linspace, matrix
    xx = linspace(0,1,40)
    plot(x,y,'o',xx,linreg.predict(matrix(xx).T),'--r')
    show()
    #We can also quantify how the model fits the original data using the mean squared error:
    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(linreg.predict(x),y))

def Correlation(data, target):
    #The best correlation measure
    #is the Pearson product-moment correlation coefficient. It's obtained by
    #dividing the covariance of the two variables by the product of their standard
    #deviations. We can compute this index between each pair of variables for
    #the iris dataset as follows:
    from numpy import corrcoef
    corr = corrcoef(data.T) # .T gives the transpose
    print(corr)
    #The function corrcoef returns a symmetric matrix of correlation coefficients
    #calculated from an input matrix in which rows are variables and columns
    #are observations. Each element of the matrix represents the correlation
    #between two variables.
    #Correlation is positive when the values increase together. It is negative
    #when one value decreases as the other increases. In particular we have
    #that 1 is a perfect positive correlation, 0 is no correlation and -1 is a perfect
    #negative correlation.
    #When the number of variables grows we can conveniently visualize the
    #correlation matrix using a pseudocolor plot:
    from pylab import pcolor, colorbar, xticks, yticks, show
    from numpy import arange
    pcolor(corr)
    colorbar() # add
    # arranging the names of the variables on the axis
    xticks(arange(0.5,4.5),['sepal length', 'sepal width', 'petal length', 'petal width'],rotation=-20)
    yticks(arange(0.5,4.5),['sepal length', 'sepal width', 'petal length', 'petal width'],rotation=-20)
    show()

def DimensionalityReduction(data, target):
    #Since the maximum number of dimensions that we can plot at the
    #same time is 3, to have a global view of the data it's necessary to embed
    #the whole data in a number of dimensions that we can visualize. This
    #embedding process is called dimensionality reduction. One of the most
    #famous techniques for dimensionality reduction is the Principal Component
    #Analysis (PCA). This technique transforms the variables of our data into
    #an equal or smaller number of uncorrelated variables called principal
    #components (PCs).
    #This time, sklearn provides us all we need to perform our analysis:
    from sklearn.decomposition import PCA
    from pylab import plot, show
    pca = PCA(n_components=2)
    #In the snippet above we instantiated a PCA object which we can use to
    #compute the first two PCs. The transform is computed as follows:
    pcad = pca.fit_transform(data)
    #And we can plot the result as usual:
    plot(pcad[target=='setosa',0],pcad[target=='setosa',1],'bo')
    plot(pcad[target=='versicolor',0],pcad[target=='versicolor',1],'ro')
    plot(pcad[target=='virginica',0],pcad[target=='virginica',1],'go')
    show()
    #The PCA projects the data into a space where the variance is maximized
    #and we can determine how much information is stored in the PCs looking
    #at the variance ratio:
    print(pca.explained_variance_ratio_)
    #Now we know that the first PC accounts for 92% of the information of the
    #original dataset while the second one accounts for the remaining 5%. We
    #can also print how much information we lost during the transformation
    #process:
    print(1-sum(pca.explained_variance_ratio_))
    #In this case we lost 2% of the information.
    #At this point, we can apply the inverse transformation to get the original
    #data back:
    data_inv = pca.inverse_transform(pcad)
    #Arguably, the inverse transformation doesn’t give us exactly the original
    #data due to the loss of information. We can estimate how much the result
    #of the inverse is likely to the original data as follows:
    print(abs(sum(sum(data - data_inv))))
    #We have that the difference between the original data and the
    #approximation computed with the inverse transform is close to zero.
    #It’s interesting to note how much information we can preserve by varying
    #the number of principal components:
    for i in range(1,5):
        pca = PCA(n_components=i)
        pca.fit(data)
        print(sum(pca.explained_variance_ratio_) * 100,'%')

def MiningNetworks():
    #use a centrality measure in order to build a meaningful visualization
    #of the data and how to find a group of nodes where the connections are
    #dense.
    #Using networkx, we can easily import the most common formats used for
    #the description of structured data:
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.read_gml('lesmiserables.gml', label='label')
    #In the code above we imported the coappearance network of characters
    #in the novel Les Miserables, freely available at https://gephi.org/datasets/
    #lesmiserables.gml.zip, in the GML format. We can also visualize the loaded
    #network with the following command: 
    nx.draw(G, with_labels=True,node_size=0,edge_color='b',alpha=.2,font_size=7)
    plt.show()
    #In this network each node represents a character of the novel and the
    #connection between two characters represents the coappearance in the
    #same chapter. It’s easy to see that the graph is not really helpful. Most of
    #the details of the network are still hidden and it’s impossible to understand
    #which are the most important nodes. In order to gain some insights about
    #our data we can study the degree of the nodes. The degree of a node is
    #considered one of the simplest centrality measures and it consists of the
    #number of connections a node has. We can summarize the degrees
    #distribution of a network looking at its maximum, minimum, median, first
    #quartile and third quartile:
    deg = nx.degree(G)
    from numpy import percentile, mean, median
    degrees = [val for (node, val) in deg]
    print(min(degrees))
    print(percentile(degrees,25)) # computes the 1st quartile
    print(median(degrees))
    print(percentile(degrees,75)) # computes the 3rd quartile
    print(max(degrees))
    #From this analysis we can decide to observe only the nodes with a degree
    #higher than 10. In order to display only those nodes we can create a new
    #graph with only the nodes that we want to visualize:
    Gt = G.copy()
    dn = nx.degree(Gt)
    GtCopy = Gt.copy()
    for n in GtCopy.nodes():
        if dn[n] <= 10:
            Gt.remove_node(n)
    nx.draw(Gt,with_labels=True,node_size=0,edge_color='b',alpha=.2,font_size=12)
    plt.show()
    #This time the graph is more readable. It makes us able to observe the most
    #relevant characters and their relationships.
    #It is also interesting to study the network through the identification of its
    #cliques. A clique is a group where a node is connected to all the other ones
    #and a maximal clique is a clique that is not a subset of any other clique
    #in the network. We can find the all maximal cliques of the our network as
    #follows:
    from networkx import find_cliques
    cliques = list(find_cliques(G))
    #And we can print the biggest clique with the following command:
    print(max(cliques, key=lambda l: len(l)))


#download data using the following Python capability
import urllib.request
from contextlib import closing
url = 'http://aima.cs.berkeley.edu/data/iris.csv'
request = urllib.request.Request(url)

#write the acquired data from the source to a csv file
with closing(urllib.request.urlopen(url)) as u, open('iris.csv', 'w') as f:
    f.write(u.read().decode('utf-8'))

#CSV can be easily parsed using the function genfromtxt of the numpy library
from numpy import genfromtxt, zeros
#read the first 4 columns
data = genfromtxt('iris.csv', delimiter=',', usecols=(0,1,2,3))
#read the fifth column
target = genfromtxt('iris.csv', delimiter=',', usecols=(4),dtype=str)

DataImportingAndVisualization(data, target)
Classification(data, target)
Clustering(data, target)
Regression(data, target)
Correlation(data, target)
DimensionalityReduction(data, target)
MiningNetworks()