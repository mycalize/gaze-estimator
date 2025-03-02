import argparse
import glob
import numpy as np
import os
import pandas as pd
from PIL import Image

"""
Create a new directory containing:
- A downsized copy of every `stride` image, filtered by `eye` and `limit`.
- A corresponding CSV file with the labels.
"""

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('src_path')
arg_parser.add_argument('dest_path')
arg_parser.add_argument('--eye', '-e', default='R')
arg_parser.add_argument('--limit', '-l', type=int, default=1)
arg_parser.add_argument('--stride', '-s', type=int, default=20)
arg_parser.add_argument('--width', type=int, default=160)
arg_parser.add_argument('--height', type=int, default=120)
args = arg_parser.parse_args()

src_dir_path = args.src_path
dest_dir_path = args.dest_path
eye = args.eye
limit = args.limit
stride = args.stride
width = args.width
height = args.height

# Create destination directory
os.mkdir(dest_dir_path)

# Read CSV file
csv_files = glob.glob(os.path.join(src_dir_path, '*.csv'))
if not csv_files: raise FileNotFoundError(f'No CSV files found in {src_dir_path}')
src_csv_file_path = csv_files[0]
labels_df = pd.read_csv(src_csv_file_path, comment='#')

# Filter by eye and limit
eye_filter = labels_df['eye'] == eye
limit_filter = (abs(np.tan(labels_df['gaze_x'])) < limit) & (abs(np.tan(labels_df['gaze_y'])) < limit)
labels_df_filtered = labels_df[eye_filter & limit_filter]

# Subsample by stride
labels_df_filtered = labels_df_filtered[np.arange(len(labels_df_filtered)) % stride == 0]

# Parse images
for i in range(len(labels_df_filtered)):
  row_idx = labels_df_filtered.index[i]
  img_filename = labels_df_filtered.loc[row_idx, 'imagefile']
  src_file_path = os.path.join(src_dir_path, img_filename)
  dest_file_path = os.path.join(dest_dir_path, img_filename)
  with Image.open(src_file_path) as im:
    im_resized = im.resize((width, height), resample=Image.Resampling.NEAREST)
    im_resized.save(dest_file_path, im_resized.format)

# Parse labels
csv_filename = f'{os.path.basename(dest_dir_path)}.csv'
dest_csv_file_path = os.path.join(dest_dir_path, csv_filename)
labels_df_filtered.to_csv(dest_csv_file_path, index=False)