import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from collections import defaultdict


def cost_fn_l1(w, x, y, lmbd):
    ''' L1 loss + L2 regularization

    w: weights to estimate d
    x: data points n x d
    y: true values n x 1
    lmbd: weight regularization

    output: loss ||x * w - y||_1 + lmbd * ||w||_2^2
    '''
    return np.abs(x @ np.expand_dims(w, 1) - y).sum() +\
           lmbd * (w ** 2).sum()

def L1LossRegression(X, Y, lmbd_reg=0.):
    ''' solves linear regression with
    L1 Loss + L2 regularization

    X: deisgn matrix n x d
    Y: true values n x 1
    lmbd_reg: weight regularization

    output: weight of linear regression d x 1
    '''
    w = minimize(cost_fn_l1, np.zeros(X.shape[1]),
                 args=(X, Y, lmbd_reg)).x
    return np.expand_dims(w, 1)


def cost_fn_l2(w, x, y, lmbd):
    ''' L2 loss + L2 regularization

    w: weights to estimate d
    x: data points n x d
    y: true values n x 1
    lmbd: weight regularization

    output: loss ||x * w - y||_2^2 + lmbd * ||w||_2^2
    '''
    
    y_hat = x @ np.expand_dims(w, 1)
    
    l2_loss = ((y_hat - y) ** 2).sum()
    l2_reg = lmbd * (w ** 2).sum()
    
    return l2_loss + l2_reg

def L2LossRegression(X, Y, lmbd_reg=0.):
    ''' solves linear regression with
    L2 Loss + L2 regularization

    X: deisgn matrix n x d
    Y: true values n x 1
    lmbd_reg: weight regularization

    output: weight of linear regression d x 1
    '''
    w = minimize(cost_fn_l2, np.zeros(X.shape[1]),
                 args=(X, Y, lmbd_reg)).x
    return np.expand_dims(w, 1)

def LeastSquares(X, Y):
    return np.linalg.solve(X.T @ X, X.T @ Y)

def RidgeRegression(X, Y, lmbd_reg):
    n = X.shape[1]
    I = np.eye(n)

    return np.linalg.solve(X.T @ X + lmbd_reg * I, X.T @ Y)

def Basis(X, k):
    ''' assume d = 1: generates the design matrix using
    orthogonal fourier basis functions
    
    X: data points n x 1
    k: maximal frequence k of the Fourier basis
    
    output: design matrix n x (2k+1) using the basis functions
    [1, cos(2 pi l x), sin(2 pi l x)]_{l=1}^k
    
    '''
    # Initialize
    design_matrix = np.ones((len(X), 2*k+1))
    
    # Fill the design matrix with the Fourier basis functions
    for l in range(1, k+1):
        design_matrix[:, 2*l-1] = np.cos(2 * np.pi * l * X).reshape(-1)
        design_matrix[:, 2*l] = np.sin(2 * np.pi * l * X).reshape(-1)
    
    return design_matrix

def FourierBasisNormalized(X, k):
    # Init
    n = X.shape[0]
    design_matrix = np.zeros((n, 2*k+1))
    # first col is constant 1/sqrt(2)
    design_matrix[:, 0] = 1/np.sqrt(2)
    
    # remaining cols are sin and cos functions
    for i in range(1, k+1):
        design_matrix[:, 2*i-1] = np.sin(2*np.pi*i*X).reshape(-1)
        design_matrix[:, 2*i] = np.cos(2*np.pi*i*X).reshape(-1)
        
    return design_matrix

def load_data(path='onedim_data.npy'):
    data = np.load(path, allow_pickle=True).item()
    
    Xtrain, Xtest = data['Xtrain'], data['Xtest']
    Ytrain, Ytest = data['Ytrain'], data['Ytest']
    
    return Xtrain, Xtest, Ytrain, Ytest

def loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def plot_data(X, Y, show=True):
    plt.scatter(X, Y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Training Data')
    if show:
        plt.show()


if __name__ == '__main__':
    
    #### Exercise 1 ####
    
    ### Part (a) ###
    X = np.array([[1, 2], [2, 3], [3, 4]])
    Y = np.array([[1], [2], [3]])
    
    w = LeastSquares(X, Y)
    print(f'Least squares: {w}', end='\n\n')
    w = RidgeRegression(X, Y, 0.1)
    print(f'Ridge regression: {w}', end='\n\n')
    
    # compare result with scipy optimize
    w = L2LossRegression(X, Y)
    print(f'Least squares scipy: {w}', end='\n\n')
    w = L2LossRegression(X, Y, 0.1)
    print(f'Ridge regression scipy: {w}', end='\n\n')
    
    
    ### Part (b) ###
    X = np.array([[1], [2], [3]])
    X_basis = Basis(X, 2)
    X_basis_normalized = FourierBasisNormalized(X, 2)
    print(X_basis)
    print(X_basis_normalized)
    
    
    ### Part (c) ###
    path = 'hw5/onedim_data.npy'
    Xtrain, Xtest, Ytrain, Ytest = load_data(path)

    # plot_data(Xtrain, Ytrain, show=False)
    
    ### Theoretical part of excerise (c) ###
    # Looking at the data, we can see that it is periodic but 
    # has some noise above the periodic function. These outliers
    # would be strongly penalized by the L2 loss function and shift the
    # regression line away from the true function. Therefore, we should
    # apply the L1 loss function to reduce the effect of the outliers.
    
    ### Empirical part of excerise (c) ###    
    ks = [1, 2, 3, 5, 10, 15, 20]
    # ks = [1, 5, 15]
    lmbds = [100, 0]
    
    basis = FourierBasisNormalized 
    # basis = Basis
    
    n_points = len(Xtrain)
    train_losses = defaultdict(list)
    test_losses = defaultdict(list)

    fig, axs = plt.subplots(len(lmbds), len(ks), figsize=(15, 10))

    for j, lmbd in enumerate(lmbds):
        print(f'λ = {lmbd}')
        
        for i, k in enumerate(ks):
            print(f'k = {k}')
            
            xs = np.linspace(0, 1, n_points)
            
            Xtrain_basis = basis(Xtrain, k)
            Xtest_basis = basis(Xtest, k)
            
            # Learn parameters w on training data
            w_k = RidgeRegression(Xtrain_basis, Ytrain, lmbd_reg=lmbd)
            
            f_k = np.dot(basis(xs, k), w_k)
            
            f_k_train = np.dot(Xtrain_basis, w_k)
            f_k_test = np.dot(Xtest_basis, w_k)
            
            train_loss = loss(Ytrain, f_k_train)
            test_loss = loss(Ytest, f_k_test)
            
            train_losses[lmbd].append(train_loss)
            test_losses[lmbd].append(test_loss)
            
            ax = axs[j, i]
            ax.scatter(Xtrain, Ytrain, alpha=0.5, s=10)
            ax.plot(xs, f_k, color='red')
            ax.set_title(f'k = {k}, λ = {lmbd}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

    plt.tight_layout()
    plt.savefig('hw5/1c_ridge_regression_fit.png')
    plt.show()

    # Plot training and test loss as a function of k for λ = 0 and λ = 30 as subplots
    fig, axs = plt.subplots(1, len(lmbds), figsize=(10, 5))
    for i, lmbd in enumerate(lmbds):
        axs[i].plot(ks, train_losses[lmbd], label='Train Loss')
        axs[i].plot(ks, test_losses[lmbd], label='Test Loss')
        axs[i].set_title(f'λ = {lmbd}')
        axs[i].set_xlabel('k')
        axs[i].set_ylabel('Loss')
        axs[i].legend()

    plt.tight_layout()
    plt.savefig('hw5/1c_ridge_regression_loss.png')
    plt.show()