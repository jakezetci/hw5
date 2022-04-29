#!/usr/bin/env python3


import argparse
import numpy as np
import pandas as pd
import json
parser = argparse.ArgumentParser()
parser.add_argument('input',  metavar='FILENAME', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    name = args.input
    data = pd.read_csv(name, names=['txt']).txt

    code = np.asarray([+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1])
    signal_code = np.repeat(code, 5)
    text_conv = np.convolve(data, signal_code[::-1])
    mean = text_conv.mean
    std = np.std(text_conv)
    text_conv[np.abs(text_conv) < 2 * std] = 0
    text_conv[text_conv > 2 * std] = 1
    text_conv[text_conv < -2 * std] = -1
    nonzero = np.nonzero(text_conv)[0]
    for i in nonzero:
        if i-1 in nonzero:
            text_conv[i-1] = 0
    binary = text_conv[text_conv != 0]
    N = binary.shape[0]
    text = (text_conv[text_conv != 0].reshape(int(N/8), 8)+1)/2
    bits = np.packbits(np.asarray(text, dtype=np.uint8))
    text_utf = ''.join(chr(i) for i in bits)

    d = {"message": text_utf}
    with open('wifi.json', 'w') as f:
        json.dump(d, f)
