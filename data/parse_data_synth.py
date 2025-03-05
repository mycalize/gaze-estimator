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
arg_parser.add_argument('--limit', '-l', type=int, default=1)
arg_parser.add_argument('--stride', '-s', type=int, default=20)
arg_parser.add_argument('--test', '-t', type=int, default=10)
arg_parser.add_argument('--width', type=int, default=160)
arg_parser.add_argument('--height', type=int, default=120)
args = arg_parser.parse_args()

src_dir_path = args.src_path
dest_dir_path = args.dest_path
limit = args.limit
stride = args.stride
test = args.test / 100
width = args.width
height = args.height

# Create destination directory and move to it
os.mkdir(dest_dir_path)

# Read CSV file
csv_files = glob.glob(os.path.join(src_dir_path, '*.csv'))
if not csv_files: raise FileNotFoundError(f'No CSV files found in {src_dir_path}')
src_csv_file_path = csv_files[0]
labels_df = pd.read_csv(src_csv_file_path, comment='#')
# List all image files in the source directory
available_images = set(os.listdir(src_dir_path))

# Filter out missing images
labels_df = labels_df[labels_df['image_L_0'].isin(available_images)]

# Split into train and test dataframes
test_gazes_df = labels_df.sample(frac=test, random_state=1)
test_gaze_x_filter = labels_df['float_gaze_x_rad'].isin(test_gazes_df['float_gaze_x_rad'])
test_gaze_y_filter = labels_df['float_gaze_y_rad'].isin(test_gazes_df['float_gaze_y_rad'])
test_filter = test_gaze_x_filter & test_gaze_y_filter
train_filter = ~test_filter
test_labels_df = labels_df[test_filter]
train_labels_df = labels_df[train_filter]

for category, df in zip(['test', 'train'], [test_labels_df, train_labels_df]):
  # Create sub-directory
  os.mkdir(os.path.join(dest_dir_path, category))

  # Subsample by stride
  df_sampled = df[np.arange(len(df)) % stride == 0]

  # Parse images
  for i in range(len(df_sampled)):
    row_idx = df_sampled.index[i]
    img_filename = df_sampled.loc[row_idx, 'image_L_0']
    src_file_path = os.path.join(src_dir_path, img_filename)
    dest_file_path = os.path.join(dest_dir_path, category, img_filename)
    if os.path.exists(src_file_path):  # Double-check existence of image file
      with Image.open(src_file_path) as im:
        im_resized = im.resize((width, height), resample=Image.Resampling.NEAREST)
        im_resized.save(dest_file_path, im.format)

  # Parse labels
  csv_filename = '04.2.csv'
  dest_csv_file_path = os.path.join(dest_dir_path, category, csv_filename)
  df_sampled.to_csv(dest_csv_file_path, index=False)
