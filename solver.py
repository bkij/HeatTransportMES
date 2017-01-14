#!/usr/bin/env python3
# TODO: python2 or python3?

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
    print("used for approximation. Must be of the form 3*4^n, n = 0,1,2,..")
    print()

def is_form_valid(num_elements):
    # check if num_elements is of form 3*4^n
    while num_elements % 4 == 0:
        num_elements /= 4
    if num_elements != 3:
        return False
    return True

def parse_arguments():
    if len(argv) < 2:
        print_usage()
        exit()
    num_elements = int(argv[1])
    if not is_form_valid(num_elements):
        print_usage()
        exit()
    return num_elements


def solve(num_elements):
    base_matrix = np.array([[2/3, -1/6, -1/3, -1/6],
                            [-1/6, 2/3, -1/6, -1/3],
                            [-1/3, -1/6, 2/3, -1/6],
                            [-1/6, -1/3, -1/6, 2/3]],
                           dtype = 'float64')
    # For each element of dimensions a1 x a2, the element's
    # matrix is cosntructed by multiplying the base matrix
    # elementwise by scalar 1 / (a1 * a2) 

def plot_heat(solution):
    pass

def main():
    num_elements = parse_arguments()
    a = solve(num_elements)
    plot_heat(a)

if __name__ == '__main__':
    main()