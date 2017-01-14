#!/usr/bin/env python3
# TODO: python2 or python3?

from sys import argv, exit
import numpy as np
import matplotlib.pyplot as plt

def print_usage():
    print("Usage: ./solver [num_elements] ")
    print()
    print("Solves a heat transport DE using the MES method")
    print("The num_elements parameter constitutes the number of finite elements")
    print("used for approximation")
    print()

def parse_arguments():
    if len(argv) < 2:
        print_usage()
        exit()
    num_elements = int(argv[1])
    return num_elements

def solve(num_elements):
    pass

def plot_heat(solution):
    pass

def main():
    num_elements = parse_arguments()
    a = solve(num_elements)
    plot_heat(a)

if __name__ == '__main__':
    main()