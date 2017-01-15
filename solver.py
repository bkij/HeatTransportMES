#!/usr/bin/env python2

from __future__ import print_function
from __future__ import division
from math import sqrt, floor
from sys import argv, exit
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# TODO: Can elements be squares only? AKA is the form 3*4^n correct
def print_usage():
    print("Usage: ./solver [num_elements] ")
    print()
    print("Solves a heat transport DE using the MES method")
    print("The num_elements parameter constitutes the number of finite elements")
    print("used for approximation. Must be of the form 3*n^2, n = 1,2,..")
    print()

def is_form_valid(num_elements):
    # check if num_elements is of form 3*n^2
    if num_elements % 3 != 0:
        return False
    num_elements /= 3
    return float(floor(sqrt(num_elements))) == sqrt(num_elements)

def parse_arguments():
    if len(argv) < 2:
        print_usage()
        exit()
    num_elements = int(argv[1])
    if not is_form_valid(num_elements):
        print_usage()
        exit()
    return num_elements

def applySquareUnit(matrix, n, a, b, c, d):
    # Creating base matrix 4x4
    p = 2./3; q = -1./6; r = -1./3
    base_matrix = np.array([[p,q,r,q],
                            [q,p,q,r],
                            [r,q,p,q],
                            [q,r,q,p]])
    # Multiply each element by 1/ (a1 * a2)
    # In our case a1 = a2 = n
    base_matrix /= (n ** 2)
    mapping = [a,b,c,d]
    for y in range(4):
        for x in range(4):
            matrix[mapping[y]][mapping[x]] += base_matrix[y][x]

def getSize(n): return n*(3*n+4)+1

def prepareCoefficientMatrix(n):
    size = getSize(n)
    matrix = np.zeros((size,size))
    # Area 1
    for y in range(n):
        for x in range(2*n):
            a = (2*n+1)*(y  )+(x  )
            b = (2*n+1)*(y  )+(x+1)
            c = (2*n+1)*(y+1)+(x+1)
            d = (2*n+1)*(y+1)+(x  )
            applySquareUnit(matrix, n, a, b, c, d)
    # Area 2
    for y in range(n):
        for x in range(n):
            offset = 2*n*(n+1)
            a = (n+1)*(y  )+(x  )+offset 
            b = (n+1)*(y  )+(x+1)+offset
            c = (n+1)*(y+1)+(x+1)+offset
            d = (n+1)*(y+1)+(x  )+offset
            applySquareUnit(matrix, n, a, b, c, d)
    which_rows = range(n*(2*n+1), n*(2*n+2)) + range(n*(2*n+2), n*(3*n+3)+1, n+1)
    for r in which_rows:
        matrix[r] = np.zeros(size)
        matrix[r][r] = 1.
    return matrix

def cbrt(x):
    k = 1
    if (x<0): k = -1
    return k*np.abs(x)**(1./3)
    
def functionG(x, y):
    r_squared = x**2+y**2
    theta = np.arctan2(y,x)
    return cbrt(r_squared)*cbrt(np.sin(theta+np.pi/2)**2)

def pairG(x1, y1, x2, y2, n): return (functionG(x1,y1)+functionG(x2,y2))/(2*n)
    
def prepareConstantMatrix(n):
    matrix = np.zeros(getSize(n))
    unit = 1./n
    half = unit/2
    # Left segment
    for i,y in zip(range(2*n+1,n*(2*n+1),2*n+1),np.linspace(1-unit,unit,n-1)):
        x = 1
        matrix[i] = pairG(x,y-half,x,y+half,n)
    # Upper left corner
    matrix[0] = pairG(-1,1-half,-1+half,1,n)
    # Upper segment
    for i,x in zip(range(1,2*n),np.linspace(-1+unit,1-unit,2*n-1)):
        y = 1
        matrix[i] = pairG(x-half,y,x+half,y,n)
    # Upper right corner
    matrix[2*n] = pairG(1,1-half,1-half,1, n)
    # Right segment !!!!!! ERROR
    for i,y in zip(range(4*n+1,n*(2*n+3),2*n+1) + range(n*(2*n+3),n*(3*n+4),2*n+1),np.linspace(1-unit,-1+unit,2*n-1)):
        x = 1
        matrix[i] = pairG(x,y-half,x,y+half,n)
    # Bottom right corner
    matrix[n*(3*n+4)] = pairG(1,-1+half,1-half,-1,n)
    # Bottom segment
    for i,x in zip(range(3*n*(n+1)+1,n*(3*n+4)),np.linspace(unit,1-unit,n-1)):
        y = 1
        matrix[i] = pairG(x-half,y,x+half,y,n)
    
    # Multiply each element by 1/ (a1 * a2)
    # In our case a1 = a2 = n
    matrix /= (n ** 2)

    return matrix
    
def prepareSolution(n):
    A = prepareCoefficientMatrix(n)
    B = prepareConstantMatrix(n)
    return np.linalg.solve(A,B)


def plot_heat(solution, n):
    # Create x, y points over the L shape domain
    X = np.array([])
    Y = np.array([])
    x_begin = -1
    y_begin = 1
    # Upper part
    for i in range(0, n + 1):
        for j in range(0, 2 * n + 1):
            Y = np.append(Y, y_begin - i * (1 / n))
            X = np.append(X, x_begin + j * (1 / n))
    # Lower part
    x_begin = 0
    y_begin = 0
    for i in range(1, n + 1):
        for j in range(0, n + 1):
            Y = np.append(Y, y_begin - i * (1 / n))
            X = np.append(X, x_begin + j * (1 / n))
    # Draw the plot

    fig = plt.figure(figsize=np.array([297,297]) / 25.4)
    ax = fig.add_subplot(2*n, 2*n, 1, projection='3d')

    # Draw points
    ax.scatter(X, Y, solution)
    # Draw lines
    for i in range(len(solution)):
        ax.plot([X[i], X[i]], [Y[i], Y[i]], zs=[0, solution[i]], color='blue')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')

    ax.set_zlim3d(0.0, 3.0)

    plt.gca().set_position([0, 0, 1, 1])
    plt.show()

def main():
    num_elements = parse_arguments()
    n = int(floor(sqrt(num_elements / 3)))
    solution = prepareSolution(n)
    plot_heat(solution, n)

if __name__ == '__main__':
    main()
