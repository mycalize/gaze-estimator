import argparse
import math
import os
import pandas as pd
import pygame
import serial
import time

PIXELS_PER_CM = 50
DIST_TO_SCREEN = 30

NUM_GAZE_PTS_X, NUM_GAZE_PTS_Y = 6, 4
PIXELS_BETWEEN_PTS = 200
W, H = NUM_GAZE_PTS_X * PIXELS_BETWEEN_PTS, NUM_GAZE_PTS_Y * PIXELS_BETWEEN_PTS
GAZE_PTS = [0, 7, 2, 9, 4, 11, 17, 22, 15, 20, 13, 18, 12, 19, 14, 21, 16, 23, 5, 10, 3, 8, 1, 6]
START_DELAY = 1000
GAZE_PT_DELAY = 1000
SPEED = 100

# Gaze pt states
CALIB = 1
START = 2
AT_GAZE_PT = 3
MOVING = 4

def get_gaze_coord(gaze_pts_idx):
  gaze_idx = GAZE_PTS[gaze_pts_idx]
  gaze_x_idx = gaze_idx % NUM_GAZE_PTS_X
  gaze_y_idx = gaze_idx // NUM_GAZE_PTS_X
  gaze_x_coord = (2 * gaze_x_idx + 1) * W // (2 * NUM_GAZE_PTS_X)
  gaze_y_coord = (2 * gaze_y_idx + 1) * H // (2 * NUM_GAZE_PTS_Y)
  return gaze_x_coord, gaze_y_coord

def get_delta_from_center(vector):
  [x, y] = vector
  delta_x = (x - W / 2) / PIXELS_PER_CM
  delta_y = (H / 2 - y) / PIXELS_PER_CM
  return delta_x, delta_y

def save_img(file_path):
  with serial.Serial('/dev/cu.usbmodem2101', 2000000) as ser, open(file_path, 'wb') as f:
    img = ser.read_until(bytes('\r\n\r\n', 'utf-8'))
    f.write(img)
    
if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('dest_path')
  args = arg_parser.parse_args()

  dest_dir_path = args.dest_path
  os.mkdir(dest_dir_path)

  pygame.init()
  screen = pygame.display.set_mode((W, H))
  clock = pygame.time.Clock()
  running = True
  dt = 0

  # Image and label state
  img_num = 0
  labels = []

  # Gaze point state
  next_gaze_pt_idx = 0
  curr_pos = pygame.Vector2((W / 2, H / 2))
  curr_col = 'blue'
  gaze_pt_state = CALIB

  # Time state
  arrive_time = pygame.time.get_ticks()

  while running:
    # Poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

    # Fill the screen with a color to wipe away anything from last frame
    screen.fill("black")

    if gaze_pt_state == CALIB:
      keys = pygame.key.get_pressed()
      if keys[pygame.K_SPACE]:
        curr_pos = pygame.Vector2(get_gaze_coord(next_gaze_pt_idx))
        curr_col = 'red'
        arrive_time = pygame.time.get_ticks()
        gaze_pt_state = START
    elif gaze_pt_state == START:
      if pygame.time.get_ticks() - arrive_time >= START_DELAY:
        arrive_time = pygame.time.get_ticks()
        gaze_pt_state = AT_GAZE_PT
    elif gaze_pt_state == AT_GAZE_PT:
      if pygame.time.get_ticks() - arrive_time >= GAZE_PT_DELAY:
        if next_gaze_pt_idx == NUM_GAZE_PTS_X * NUM_GAZE_PTS_Y - 1:
          running = False
        next_gaze_pt_idx += 1
        gaze_pt_state = MOVING
    elif gaze_pt_state == MOVING:
      dest_pos = pygame.Vector2(get_gaze_coord(next_gaze_pt_idx))
      direction = dest_pos - curr_pos
      distance = direction.length()

      if distance <= SPEED * dt:
        curr_pos = dest_pos
        arrive_time = pygame.time.get_ticks()
        gaze_pt_state = AT_GAZE_PT
      else:
        direction.scale_to_length(SPEED * dt)
        curr_pos += direction

    pygame.draw.circle(screen, curr_col, curr_pos, 20)

    # flip() the display to put your work on screen
    pygame.display.flip()

    if gaze_pt_state == AT_GAZE_PT or gaze_pt_state == MOVING:
      # Save img and label
      img_name = f'{img_num:>06}.jpg'
      img_num += 1
      delta_x, delta_y = get_delta_from_center(curr_pos)
      gaze_x = round(math.atan(delta_x / DIST_TO_SCREEN), 6)
      gaze_y = round(math.atan(delta_y / DIST_TO_SCREEN), 6)
      label = {
        'imagefile': img_name,
        'eye': 'R',
        'gaze_x': gaze_x,
        'gaze_y': gaze_y,
      }
      img_file_path = os.path.join(dest_dir_path, img_name)
      save_img(img_file_path)
      labels.append(label)

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick() / 1000

  # Save labels as CSV file
  labels_df = pd.DataFrame(labels)
  csv_filename = f'{os.path.basename(dest_dir_path)}.csv'
  csv_file_path = os.path.join(dest_dir_path, csv_filename)
  labels_df.to_csv(csv_file_path, index=False)

  pygame.quit()