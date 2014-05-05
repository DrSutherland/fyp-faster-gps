import numpy as np

# Phase taps according to the ICD-GPS-200 specification
PHASE_TAPS = np.array([
  [2,6],  # PRN 1
  [3,7],  # PRN 2
  [4,8],  # PRN 3
  [5,9],  # PRN 4
  [1,9],  # PRN 5
  [2,10], # PRN 6
  [1,8],  # PRN 7
  [2,9],  # PRN 8
  [3,10], # PRN 9
  [2,3]   # PRN 10
]) - 1 # Minus 1 to take into account Python's zero-indexed arrays

def get_phase_taps(prn):
  return PHASE_TAPS[prn-1]

def generate(prn):
  # G1 = 1 + x3 + x10
  poly1 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

  # G2 = 1 + x2 + x3 + x6 + x8 + x9 + x10
  poly2 = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])

  # Allocate G1 (shift register)
  sr1 = np.ones(poly1.size)

  # Allocate G2 (shift register)
  sr2 = np.ones(poly2.size)

  # Get phase tap indices for selected PRN
  phase_taps = get_phase_taps(prn=prn)

  # G3 (sr1) length is ((Size of 2^G1) - 1)
  L = (2 ** sr1.size) - 1

  # Allocate G3 (output)
  G  = np.zeros(L)

  # Generate C/A code sequence
  for i in range(L):
    tapped = np.mod(np.sum(sr2[phase_taps]), 2)

    G[i] = np.mod(sr1[-1] + tapped, 2);

    sr1[-1] = np.mod(np.sum(sr1 * poly1), 2)
    sr1 = np.roll(sr1, 1)

    sr2[-1] = np.mod(np.sum(sr2 * poly2), 2)
    sr2 = np.roll(sr2, 1)

  return G

if __name__ == '__main__':
  print generate(prn=1)[0:10]
