#!/usr/bin/env python3

"""A program for creating Nonograms (1) from an image.

(1) http://gambiter.com/puzzle/Nonogram.html

Author(s): Kunaal Verma (https://github.com/vermakun)
Revision: 0.2.1 - 2022.02.15
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2 as cv

def main(arguments):

    args = argument_parse(arguments)
    outdir_check(args.output)

    for path, folders, files in os.walk(args.input):
        for file in files:
            filepath = os.path.join(path, file)
            root, ext = os.path.splitext(file)

            if not ext == '.png':
                continue

            nonogrid = create_nono_grid(filepath, args.debug)

            # Produce dataframe
            df = pd.DataFrame(nonogrid)

            # Save the dataframe as a csv file
            df.to_csv(os.path.join(args.output, root + '.csv'), index = False, header = False)

def argument_parse(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input',  default='.',   help='Input file/directory')
    parser.add_argument('-o', '--output', default='csv', help="Output file")
    parser.add_argument('-d', '--debug',  default=False, help="Print debug info")
    args = parser.parse_args(arguments)
    return args

def outdir_check(outpath):
    if os.path.isdir(outpath):
        for root, dirs, files in os.walk(outpath, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        os.mkdir(outpath)

def create_nono_grid(filepath, debug):

    # Read image
    img = cv.imread(filepath, cv.IMREAD_UNCHANGED)

    # Original image
    if debug: cv.imshow('orig', img), cv.waitKey()

    # Black/White threshold
    bw_img = img[:,:,3] & 255 - img[:,:,0]

    # Display B/W image (w/ alpha detail)
    if debug: cv.imshow('bw',bw_img), cv.waitKey()

    # Extract channels
    samp = bw_img

    # Image dimensions
    row, col = samp.shape

    # Produce row numbers
    row_results = []

    for i in np.arange(row):
      
      cur_row = np.array([])
      cur_cnt = 0

      for j in np.arange(col):
        if samp[i,j] > 0:
          cur_cnt = cur_cnt + 1
        else:
          cur_cnt = 0
        cur_row = np.append(cur_row,cur_cnt)
      
      cur_row = cur_row[cur_row != 0]
      cur_val = cur_row
      if debug: print(cur_row)
      cur_row = np.diff(np.insert(cur_row,0,0))
      if debug: print(cur_row)
      cur_row = np.diff(np.append(cur_row,-1))
      if debug: print(cur_row)
      if debug: print(cur_val[cur_row < 0])
      if debug: print(' ')
      row_results.append(cur_val[cur_row < 0].tolist())

    # Produce column numbers
    col_results = []

    for i in np.arange(col):

      cur_col = np.array([])
      cur_cnt = 0

      for j in np.arange(row):

        if samp[j,i] > 0:
          cur_cnt = cur_cnt + 1
        else:
          cur_cnt = 0
        cur_col = np.append(cur_col,cur_cnt)

      cur_col = cur_col[cur_col != 0]
      cur_val = cur_col
      if debug: print(cur_col)
      cur_col = np.diff(np.insert(cur_col,0,0))
      if debug: print(cur_col)
      cur_col = np.diff(np.append(cur_col,-1))
      if debug: print(cur_col)
      if debug: print(cur_val[cur_col < 0])
      if debug: print(' ')
      col_results.append(cur_val[cur_col < 0].tolist())

    # Trim empty rows and cols from ends
    while not row_results[0]:
      row_results = row_results[1::]

    while not row_results[-1]:
      row_results = row_results[0:-1]

    while not col_results[0]:
      col_results = col_results[1::]

    while not col_results[-1]:
      col_results = col_results[0:-1]

    if debug: print(row_results)
    if debug: print(col_results)

    # Determine puzzle padding width

    row_heights = [len(i) for i in row_results]
    if debug: print(row_heights)
    max_row_width = max(row_heights)
    if debug: print(max_row_width)

    col_heights = [len(i) for i in col_results]
    if debug: print(col_heights)
    max_col_width = max(col_heights)
    if debug: print(max_col_width)

    max_width = np.max([max_row_width, max_col_width])
    if debug: print(max_width)

    # Generate puzzle matrix

    mw = max_width

    output = np.empty((mw + row, mw + col,))
    output[:] = np.nan

    # Add row number elements
    ic = 0
    for i in row_results:
      
      jc = 0
      for j in i:

        output[mw+ic, mw-jc-1] = int(j)
        
        jc = jc + 1
      
      ic = ic + 1

    # Add column number elements
    ic = 0
    for i in col_results:
      
      jc = 0
      for j in i:

        output[mw-jc-1, mw+ic] = int(j)
        
        jc = jc + 1
      
      ic = ic + 1

    if debug: print('Sample Output:')
    if debug: print(output[0:15,0:15])

    return output

if __name__ == '__main__':
    main(sys.argv[1:])