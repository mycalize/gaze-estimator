import argparse
import csv
import glob
import os
from PIL import Image

"""
Create a new directory containing:
- A downsized copy of every `stride` image.
- A corresponding CSV file with the labels.
"""

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('src_path')
arg_parser.add_argument('dest_path')
arg_parser.add_argument('--stride', '-s', type=int, default=20)
arg_parser.add_argument('--width', type=int, default=160)
arg_parser.add_argument('--height', type=int, default=120)
arg_parser.add_argument('--csv_nrows_skip', type=int, default=11)
args = arg_parser.parse_args()

src_dir_path = args.src_path
dest_dir_path = args.dest_path
stride = args.stride
width = args.width
height = args.height
csv_nrows_skip = args.csv_nrows_skip

# Create destination directory
os.mkdir(dest_dir_path)

filenames = sorted(os.listdir(src_dir_path))
jpg_filenames = [filename for filename in filenames if filename.endswith('jpg')]

# Parse images
for i, filename in enumerate(jpg_filenames):
  if i % stride == 0:
    src_file_path = os.path.join(src_dir_path, filename)
    dest_file_path = os.path.join(dest_dir_path, filename)
    with Image.open(src_file_path) as im:
      im_resized = im.resize((width, height), resample=Image.Resampling.NEAREST)
      im_resized.save(dest_file_path, im_resized.format)

# Parse labels
csv_files = glob.glob(os.path.join(src_dir_path, '*.csv'))
if not csv_files: raise FileNotFoundError(f'No CSV files found in {src_dir_path}')
src_csv_file_path = csv_files[0]

csv_filename = f'{os.path.basename(dest_dir_path)}.csv'
dest_csv_path = os.path.join(dest_dir_path, csv_filename)
with open(src_csv_file_path, newline='') as src_csv, open(dest_csv_path, 'w', newline='') as dest_csv:
  reader = csv.reader(src_csv)
  writer = csv.writer(dest_csv)
  # Skip comments rows in CSV
  for _ in range(csv_nrows_skip):
    next(reader)
  # Write header row
  writer.writerow(next(reader))
  for i, row in enumerate(reader):
    if i % stride == 0:
      writer.writerow(row)