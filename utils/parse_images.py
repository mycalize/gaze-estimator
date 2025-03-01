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
arg_parser.add_argument('--width', type=int, default=160)
arg_parser.add_argument('--height', type=int, default=120)
args = arg_parser.parse_args()

src_dir_path = args.src_path
dest_dir_path = args.dest_path
stride = args.stride
width = args.width
height = args.height

# Create destination directory
os.mkdir(dest_dir_path)

filenames = sorted(os.listdir(src_dir_path))
for i, filename in enumerate(filenames):
  if filename.endswith('.jpg') and i % stride == 0:
    src_file_path = os.path.join(src_dir_path, filename)
    dest_file_path = os.path.join(dest_dir_path, filename)
    with Image.open(src_file_path) as im:
      im_resized = im.resize((width, height), resample=Image.Resampling.NEAREST)
      im_resized.save(dest_file_path, im_resized.format)
