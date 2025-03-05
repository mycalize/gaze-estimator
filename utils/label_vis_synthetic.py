import matplotlib.pyplot as plt
import numpy as np

def get_labels(filename):
  """ Load gaze_x and gaze_y from one CSV file """
  gaze_labels = np.loadtxt(filename, delimiter=',', skiprows=12, usecols=(7, 8))
  return gaze_labels

def show_labels_hist(filename, n_bins=20):
  """ Generate a histogram for gaze_x and gaze_y """
  labels = get_labels(filename)
  gaze_x, gaze_y = labels[:, 0], labels[:, 1]

  mean_x, mean_y = np.mean(labels, axis=0)
  std_x, std_y = np.std(labels, axis=0)
  min_x, min_y = np.min(labels, axis=0)
  max_x, max_y = np.max(labels, axis=0)

  _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True)

  ax1.hist(gaze_x, bins=n_bins, color='lightblue')
  ax1.set_xlabel('gaze_x')
  ax1.set_ylabel('Frequency')
  ax1_stats = f'mean = {mean_x:.4f}\nstd = {std_x:.4f}\nmin = {min_x:.4f}\nmax = {max_x:.4f}'
  ax1.text(0.05, 0.95, ax1_stats, transform=ax1.transAxes, va='top')

  ax2.hist(gaze_y, bins=n_bins, color='lightblue')
  ax2.set_xlabel('gaze_y')
  ax2_stats = f'mean = {mean_y:.4f}\nstd = {std_y:.4f}\nmin = {min_y:.4f}\nmax = {max_y:.4f}'
  ax2.text(0.05, 0.95, ax2_stats, transform=ax2.transAxes, va='top')

  plt.show()