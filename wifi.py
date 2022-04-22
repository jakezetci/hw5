#!/usr/bin/env python3


import argparse


parser = argparse.ArgumentParser()
parser.add_argument('input',  metavar='FILENAME', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args.input)
