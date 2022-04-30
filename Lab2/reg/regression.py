from turtle import color
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import White
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    
    delta = 0.025
    x = np.arange(-1.0, 1.0, delta)
    y = np.arange(-1.0, 1.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-1*X**2/(2*beta))*np.exp(-1*Y**2/(2*beta))/(2*np.pi*beta)

    plt.contour(X, Y, Z, colors='b')
    plt.title('Prior Distribution of a = [a0 a1]')
    plt.xlabel('a0')
    plt.xlim([-1,1])
    plt.ylabel('a1')
    plt.ylim([-1,1])
    plt.plot(-0.1, -0.5, 'ro')
    plt.savefig("prior.pdf")
    plt.show()
    
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here
    V0 = beta*np.identity(2)
    ones = np.ones((len(x), 1))
    X = np.concatenate((ones, x), axis=1)
    VN = sigma2 * np.linalg.inv(sigma2*np.linalg.inv(V0) + np.dot(X.T, X))
    wn = np.dot(np.dot(VN, X.T), z)/sigma2

    delta = 0.01
    xval = np.arange(-1.0, 1.0, delta)
    yval = np.arange(-1.0, 1.0, delta)
    X, Y = np.meshgrid(xval, yval)

    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = util.density_Gaussian(wn.reshape(2),VN,np.array([[X[i,j],Y[i,j]]]))[0]

    plt.contour(X, Y, Z, colors='b')
    plt.xlabel('a0')
    plt.xlim([-1,1])
    plt.ylabel('a1')
    plt.ylim([-1,1])
    plt.plot(-0.1, -0.5, 'ro')

    print(x)

    if len(x) == 1:
        plt.title('Posterior Distribution of a given 1 training sample')
        plt.savefig("posterior1.pdf")
    elif len(x) == 5:
        plt.title('Posterior Distribution of a given 5 training samples')
        plt.savefig("posterior5.pdf")
    elif len(x) == 100:
        plt.title('Posterior Distribution of a given 100 training samples')
        plt.savefig("posterior100.pdf")
    plt.show()

    return (wn, VN)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here

    predictions, errors = [], []
    for i in range(len(x)):
        input = np.array([[1], [x[i]]])
        pred = np.dot(mu.T, input)[0,0] 
        sd = sigma2 + np.dot(np.dot(input.T, Cov), input)[0,0]
        print("%d\n\n\n", sd)
        predictions.append(pred)
        errors.append(sd**0.5)

    
    plt.scatter(x_train, z_train, c="red")
    plt.scatter(x, predictions, c="blue")
    plt.errorbar(x, predictions, errors)
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    if len(x_train) == 1:
        plt.title('Predictions based on 1 training sample')
        plt.savefig("predict1.pdf")
    elif len(x_train) == 5:
        plt.title('Predictions based on 5 training samples')
        plt.savefig("predict5.pdf")
    elif len(x_train) == 100:
        plt.title('Predictions based on 100 training samples')
        plt.savefig("predict100.pdf")
    plt.show()

    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns_list  = [1, 5, 100]
    
    # prior distribution p(a)
    priorDistribution(beta)

    for ns in ns_list:
        # used samples
        x = x_train[0:ns]
        z = z_train[0:ns]
        
        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x,z,beta,sigma2)
        
        # distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
