import math

def convert_labels(label, x_min=-1, x_max=1, y_min=-1, y_max=1, dims=(4,4)):
  """ Calculate class in a grid of classes from gaze
  vectors. Class 0 denotes the class with smallest
  horizontal/vertical gaze angles. """
  x_width = (x_max - x_min) / dims[0]
  y_width = (y_max - y_min) / dims[1]

  class_x = (math.tan(label[0]) - x_min) // x_width
  class_y = (math.tan(label[1]) - y_min) // y_width
  class_label = dims[1] * class_y + class_x

  return int(class_label)