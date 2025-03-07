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
arg_parser.add_argument('--test', '-t', type=int, default=10)
arg_parser.add_argument('--val', '-v', type=int, default=10)
arg_parser.add_argument('--width', type=int, default=160)
arg_parser.add_argument('--height', type=int, default=120)
args = arg_parser.parse_args()

src_dir_path = args.src_path
dest_dir_path = args.dest_path
eye = args.eye
limit = args.limit
stride = args.stride
test = args.test / 100
val = args.val / 100
width = args.width
height = args.height

# Create destination directory and move to it
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

# Split into train, val, and test gaze points
unique_gazes_df = labels_df_filtered.drop_duplicates(subset=['gaze_x', 'gaze_y'])
test_val_gazes_df = unique_gazes_df.sample(frac=test + val, random_state=1)
num_test_gazes = int(test * len(unique_gazes_df))
test_gazes_df = test_val_gazes_df.iloc[:num_test_gazes]
val_gazes_df = test_val_gazes_df.iloc[num_test_gazes:]

# Split into test dataframe
test_x_filter = labels_df_filtered['gaze_x'].isin(test_gazes_df['gaze_x'])
test_y_filter = labels_df_filtered['gaze_y'].isin(test_gazes_df['gaze_y'])
test_filter = test_x_filter & test_y_filter
test_labels_df = labels_df_filtered[test_filter]

# Split into val dataframe
val_x_filter = labels_df_filtered['gaze_x'].isin(val_gazes_df['gaze_x'])
val_y_filter = labels_df_filtered['gaze_y'].isin(val_gazes_df['gaze_y'])
val_filter = val_x_filter & val_y_filter
val_labels_df = labels_df_filtered[val_filter]

# Split into train dataframe
train_filter = ~(test_filter | val_filter)
train_labels_df = labels_df_filtered[train_filter]

for category, df in zip(['test', 'val', 'train'], [test_labels_df, val_labels_df, train_labels_df]):
  # Create sub-directory
  os.mkdir(os.path.join(dest_dir_path, category))

  # Subsample by stride
  df_sampled = df[np.arange(len(df)) % stride == 0]

  # Parse images
  for i in range(len(df_sampled)):
    row_idx = df_sampled.index[i]
    img_filename = df_sampled.loc[row_idx, 'imagefile']
    src_file_path = os.path.join(src_dir_path, img_filename)
    dest_file_path = os.path.join(dest_dir_path, category, img_filename)
    with Image.open(src_file_path) as im:
      im_resized = im.resize((width, height), resample=Image.Resampling.NEAREST)
      im_resized.save(dest_file_path, im_resized.format)

  # Parse labels
  csv_filename = f'{os.path.basename(dest_dir_path)}_{category}.csv'
  dest_csv_file_path = os.path.join(dest_dir_path, category, csv_filename)
  df_sampled.to_csv(dest_csv_file_path, index=False)