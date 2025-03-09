def check_disjoint(d1, d2):
  d1_labels = [tuple(d1[i][1].tolist()) for i in range(len(d1))]
  d2_labels = [tuple(d2[i][1].tolist()) for i in range(len(d2))]
  return set(d1_labels).isdisjoint(set(d2_labels))