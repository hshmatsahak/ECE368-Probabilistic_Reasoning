from matplotlib import mlab
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    male_count = 0
    female_count = 0

    mu_male, mu_female = np.array([0, 0]), np.array([0, 0])
    cov, cov_male, cov_female = np.array([[0,0], [0,0]]), np.array([[0,0], [0,0]]), np.array([[0,0], [0,0]])

    for i in range(x.shape[0]):
        if y[i] == 1:
            male_count += 1
            mu_male += x[i, :]
        else:
            female_count += 1
            mu_female += x[i, :]

    mu_male = np.divide(mu_male, male_count)
    mu_female = np.divide(mu_female, female_count)

    for i in range(x.shape[0]):
        data = np.array(x[i,:])
        if y[i] == 1:
            data = data - mu_male
            cov_male = np.add(cov_male, np.outer(data, data))
        else:
            data = data - mu_female
            cov_female = np.add(cov_female, np.outer(data, data))
        cov = np.add(cov, np.outer(data, data))

    cov_male = np.divide(cov_male, male_count)
    cov_female = np.divide(cov_female, female_count)
    cov = np.divide(cov, male_count+female_count)

    # print(mu_male)
    # print(mu_female)
    # print(cov)
    # print(cov_male)
    # print(cov_female)

    return (mu_male,mu_female,cov,cov_male,cov_female)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    num_samples = x.shape[0]

    lda_correct = 0
    lda_wrong = 0
    lda_male = util.density_Gaussian(mu_male, cov, x)
    lda_female = util.density_Gaussian(mu_female, cov, x)

    lda_male_pred = lda_male > lda_female
    
    for i in range(lda_male_pred.shape[0]):
        if (lda_male_pred[i] and y[i]==1) or (not lda_male_pred[i] and y[i]==2):
            lda_correct+=1
        else:
            lda_wrong+=1

    assert(lda_correct+lda_wrong==num_samples)

    mis_lda = lda_wrong/num_samples

    qda_correct = 0
    qda_wrong = 0
    qda_male = util.density_Gaussian(mu_male, cov_male, x)
    qda_female = util.density_Gaussian(mu_female, cov_female, x)

    qda_male_pred = qda_male > qda_female
    
    for i in range(qda_male_pred.shape[0]):
        if (qda_male_pred[i] and y[i]==1) or (not qda_male_pred[i] and y[i]==2):
            qda_correct+=1
        else:
            qda_wrong+=1

    assert(qda_correct+qda_wrong==num_samples)

    mis_qda = qda_wrong/num_samples
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    inv_Sigma = np.linalg.inv(cov)
    beta_male = np.dot(mu_male,inv_Sigma)
    gamma_male = -0.5*np.dot(beta_male, mu_male.T)
    beta_female = np.dot(mu_female,inv_Sigma)
    gamma_female = -0.5*np.dot(beta_female, mu_female.T)

    m = beta_male - beta_female
    b = gamma_male - gamma_female

    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    print(f'LDA misclassification rate: {mis_LDA}\nQDA misclassification rate: {mis_QDA}')
    
    heights = np.linspace(50, 80, 100)   
    weights = np.linspace(80, 280, 100)   
    H, W = np.meshgrid(heights, weights)

    male_lda = []
    male_qda = []
    female_lda = []
    female_qda = []

    for weight in weights:
        male_lda_tmp = []
        male_qda_tmp = []
        female_lda_tmp = []
        female_qda_tmp = []
        for height in heights:
            sample = np.array([[height, weight]])
            male_lda_tmp.extend(util.density_Gaussian(mu_male,cov,sample))
            male_qda_tmp.extend(util.density_Gaussian(mu_male,cov_male,sample))
            female_lda_tmp.extend(util.density_Gaussian(mu_female,cov,sample))
            female_qda_tmp.extend(util.density_Gaussian(mu_female,cov_female,sample))
        male_lda.append(male_lda_tmp[:])
        male_qda.append(male_qda_tmp[:])
        female_lda.append(female_lda_tmp[:])
        female_qda.append(female_qda_tmp[:])

    # LDA
    for i in range(x_train.shape[0]):
        if y_train[i]==1:
            col = 'green' # male
        else:
            col = 'blue' # female
        plt.scatter(x_train[i, 0], x_train[i, 1], c = col, s = 10, linewidth = 0)  

    plt.contour(H,W,male_lda,colors='g')
    plt.contour(H,W,female_lda,colors='b')

    lda_decision = np.asarray(male_lda) - np.asarray(female_lda)
    plt.contour(H,W,lda_decision,0,colors='r')
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.title('LDA Visualization')
    plt.savefig("lda.pdf")
    plt.show()

    # QDA
    for i in range(x_train.shape[0]):
        if y_train[i]==1:
            col = 'green' # male
        else:
            col = 'blue' # female
        plt.scatter(x_train[i, 0], x_train[i, 1], c = col, s = 10, linewidth = 0)  

    plt.contour(H,W,male_qda,colors='g')
    plt.contour(H,W,female_qda,colors='b')

    qda_decision = np.asarray(male_qda) - np.asarray(female_qda)
    plt.contour(H,W,qda_decision,0,colors='r')
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.title('QDA Visualization')
    plt.savefig("qda.pdf")
    plt.show()