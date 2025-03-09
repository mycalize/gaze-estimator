import argparse
import glob
import numpy as np
import os
import pandas as pd
from PIL import Image

"""
Eye Gaze Dataset Preprocessing Script

This script processes raw eye gaze datasets into a standardized format
for training gaze estimation models.It takes a source directory containing images and a
CSV file with gaze labels. The script expects aCSV file in the source directory with the 
following columns:

1. 'imagefile': Filename of the corresponding eye image in the same source directory

2. 'eye': Indicator of which eye the data corresponds to(e.g., 'R' for right eye, 'L' for left eye)

3. 'gaze_x': The horizontal gaze angle in radians
   - Positive values indicate looking right
   - Negative values indicate looking left

4. 'gaze_y': The vertical gaze angle in radians
   - Positive values indicate looking up
   - Negative values indicate looking down

The CSV file may include a header and can have comment lines starting with '#'.

Example:
-----------------------------------------
# Comment
imagefile,eye,gaze_x,gaze_y
01.jpg,R,0.2,-0.1
02.jpg,R,0.3,-0.2
03.jpg,L,0.25,-0.15
...
-----------------------------------------

It creates a new organized dataset with the following structure:

dest_path/
├── train/
│   ├── images...
│   └── [dataset_name]_train.csv
├── val/
│   ├── images...
│   └── [dataset_name]_val.csv
└── test/
    ├── images...
    └── [dataset_name]_test.csv

Key features:
- Filters data by specified eye (left or right) and gaze angle limits
- Splits dataset into train/validation/test sets based on unique gaze points (not random images)
- Subsamples data at specified stride to manage dataset size
- Resizes all images to consistent dimensions
- Creates corresponding CSV files with labels for each split

Usage:
  python parse_data.py src_path dest_path [options]

Arguments:
  src_path             Source directory containing images and CSV file
  dest_path            Destination directory for processed dataset

Options:
  --eye, -e            Eye to use (default: 'R' for right eye)
  --limit, -l          Limit for filtering: abs(tan(gaze_angle)) < limit, negative means off (default: -1)
  --stride, -s         Stride for subsampling images (default: 20)
  --test, -t           Percentage of data for test set (default: 10%)
  --val, -v            Percentage of data for validation set (default: 10%)
  --crop               Crop image into square, if set, only accepts the height option
  --width              Target image width in pixels (default: 120)
  --height             Target image height in pixels (default: 120)
"""

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('src_path')
arg_parser.add_argument('dest_path')
arg_parser.add_argument('--eye', '-e', default='R')
arg_parser.add_argument('--limit', '-l', type=int, default=-1)
arg_parser.add_argument('--stride', '-s', type=int, default=20)
arg_parser.add_argument('--test', '-t', type=int, default=10)
arg_parser.add_argument('--val', '-v', type=int, default=10)
arg_parser.add_argument('--crop', type=bool, default=True)
arg_parser.add_argument('--width', type=int, default=120)
arg_parser.add_argument('--height', type=int, default=120)
args = arg_parser.parse_args()

src_dir_path = args.src_path
dest_dir_path = args.dest_path
eye = args.eye
limit = args.limit
stride = args.stride
test = args.test / 100
val = args.val / 100
crop = args.crop
width = args.width
height = args.height

# Create destination directory and move to it
os.mkdir(dest_dir_path)

# Read CSV file
csv_files = glob.glob(os.path.join(src_dir_path, '*.csv'))
if not csv_files:
    raise FileNotFoundError(f'No CSV files found in {src_dir_path}')
src_csv_file_path = csv_files[0]
labels_df = pd.read_csv(src_csv_file_path, comment='#')

# Filter by eye and limit
eye_filter = labels_df['eye'] == eye
labels_df_filtered = labels_df[eye_filter]
if limit >= 0:
    limit_filter = (abs(np.tan(labels_df['gaze_x'])) < limit) & (abs(np.tan(labels_df['gaze_y'])) < limit)
    labels_df_filtered = labels_df_filtered[limit_filter]

# Split into train, val, and test gaze points
unique_gazes_df = labels_df_filtered.drop_duplicates(
    subset=['gaze_x', 'gaze_y'])
test_val_gazes_df = unique_gazes_df.sample(frac=test + val, random_state=1)
num_test_gazes = int(test * len(unique_gazes_df))
test_gazes_df = test_val_gazes_df.iloc[:num_test_gazes]
val_gazes_df = test_val_gazes_df.iloc[num_test_gazes:]

# Split into test dataframe
test_labels_df = pd.merge(
    labels_df_filtered,
    test_gazes_df[['gaze_x', 'gaze_y']],
    on=['gaze_x', 'gaze_y'],
    how='inner'
)

# Split into val dataframe
val_labels_df = pd.merge(
    labels_df_filtered,
    val_gazes_df[['gaze_x', 'gaze_y']],
    on=['gaze_x', 'gaze_y'],
    how='inner'
)

# Split into train dataframe
test_val_labels_df = pd.concat([test_labels_df, val_labels_df])
train_filter = ~(labels_df_filtered['imagefile'].isin(test_val_labels_df['imagefile']))
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
            if crop:
                old_width, old_height = im.size
                crop_width = min(old_width, old_height)
                left = (old_width - crop_width) / 2
                upper = (old_height - crop_width) / 2
                right = (old_width + crop_width) / 2
                lower = (old_height + crop_width) / 2

                im_cropped = im.crop((left, upper, right, lower))
                im_resized = im_cropped.resize((width, width), resample=Image.Resampling.NEAREST)
            else:
                im_resized = im.resize((width, height), resample=Image.Resampling.NEAREST)
            im_resized.save(dest_file_path, im_resized.format)

    # Parse labels
    csv_filename = f'{os.path.basename(dest_dir_path)}_{category}.csv'
    dest_csv_file_path = os.path.join(dest_dir_path, category, csv_filename)
    df_sampled.to_csv(dest_csv_file_path, index=False)
