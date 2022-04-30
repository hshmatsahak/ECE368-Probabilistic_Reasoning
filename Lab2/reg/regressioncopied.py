import numpy as np
import matplotlib.pyplot as plt
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
    #plot gradient
    plt.plot([-0.1], [-0.5], marker='o', markersize=6, color='orange')
    a0_grid = np.linspace(-1, 1, 100)   
    a1_grid = np.linspace(-1, 1, 100)   
    A0, A1 = np.meshgrid(a0_grid, a1_grid)
    a0 = A0[0].reshape(100, 1)
    Gaussian_contour = []
    #make data points samples to pass into density_Gaussian
    for i in range(0, 100):
      samples = np.concatenate((a0, A1[i].reshape(100, 1)), 1)
      Gaussian_contour.append(util.density_Gaussian([0, 0],[[beta, 0], [0, beta]], samples))
    
    # plot the contours
    plt.contour(A0, A1, Gaussian_contour, colors='b')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title('prior distribution of a0 and a1')
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
    cov_a_inverse = [[1/beta, 0], [0, 1/beta]]
    A = np.append(np.ones(shape=(len(x), 1)), x, axis=1)
    cov_w_inverse = 1/sigma2
    mu = np.linalg.inv((cov_a_inverse + cov_w_inverse * np.dot(A.T, A))) @ (cov_w_inverse * np.dot(A.T, z))
    mu = mu.reshape(2, 1).squeeze()
    Cov = np.linalg.inv(cov_a_inverse + np.dot(A.T, np.dot(cov_w_inverse, A)))

    a0_grid = np.linspace(-1, 1, 100)   
    a1_grid = np.linspace(-1, 1, 100)

    A0, A1 = np.meshgrid(a0_grid, a1_grid)

    a0 = A0[0].reshape(100, 1)
    posterior_contour = []

    #make data points samples to pass into density_Gaussian
    for i in range(0, 100):
      samples = np.concatenate((a0, A1[i].reshape(100, 1)), 1)
      posterior_contour.append(util.density_Gaussian(mu.T, Cov, samples))
    
    # plot the contours
    plt.contour(A0, A1, posterior_contour, colors='blue')
    plt.plot([-0.1], [-0.5], marker='o', markersize=6, color='orange')
    plt.xlabel('a0')
    plt.ylabel('a1')
    if (len(x) == 1):
        plt.title('posterior distribution based on 1 data sample')
        plt.savefig("posterior1.pdf")
    elif (len(x) == 5):
        plt.title('posterior distribution based on 5 data samples')
        plt.savefig("posterior5.pdf")
    elif (len(x) == 100):
        plt.title('posterior distribution based on 100 data samples')
        plt.savefig("posterior100.pdf")
    plt.show()
    return (mu,Cov)

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
    cov_a_inverse = [[1/beta, 0], [0, 1/beta]]
    A = np.append(np.ones(shape=(len(x), 1)), np.expand_dims(x, 1), axis=1)
    cov_w = sigma2

    mu_z = np.dot(A, mu)
    cov_z = cov_w + np.dot(A, np.dot(Cov, A.T))
    std_z = np.sqrt(np.diag(cov_z))
    print(mu_z)
    print(cov_z)
    print(std_z)
    
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('x')
    plt.ylabel('z')
    plt.scatter(x_train, z_train, color = 'blue')
    plt.errorbar(x, mu_z, yerr=std_z, fmt='ro')

    if (len(x_train) == 1):
        plt.title('prediction based on 1 data sample')
        plt.savefig("predict1.pdf")
    elif (len(x_train) == 5):
        plt.title('prediction based on 5 data samples')
        plt.savefig("predict5.pdf")
    elif (len(x_train) == 100):
        plt.title('prediction based on 100 data samples')
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
    ns  = 1
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)

    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)

    # number of training samples used to compute posterior
    ns  = 5
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)

    # number of training samples used to compute posterior
    ns  = 100
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    
    # distribution of the prediction
    # predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)