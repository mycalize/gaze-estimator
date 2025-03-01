import argparse
import os
from PIL import Image

"""
Create a new directory containing every `stride` image and downsized.
"""

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('src_path')
arg_parser.add_argument('dest_path')
arg_parser.add_argument('--stride', '-s', type=int, default=20)
args = arg_parser.parse_args()

src_dir_path = args.src_path
dest_dir_path = args.dest_path
stride = args.stride

# Create destination directory
os.mkdir(dest_dir_path)

for i, filename in enumerate(os.listdir(src_dir_path)):
  if i % stride == 0:
    src_file_path = os.path.join(src_dir_path, filename)
    dest_file_path = os.path.join(dest_dir_path, filename)
    with Image.open(src_file_path) as im:
      im_resized = im.resize((160, 120), resample=Image.Resampling.NEAREST)
      im_resized.save(dest_file_path, im_resized.format)
