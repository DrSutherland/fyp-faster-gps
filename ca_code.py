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

# G1 = 1 + x3 + x10
POLYNOMIAL_1 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

# G2 = 1 + x2 + x3 + x6 + x8 + x9 + x10
POLYNOMIAL_2 = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])

# Shift register size
SR_SIZE = 10

def get_phase_taps(prn):
  return PHASE_TAPS[prn-1]

def generate(prn):
  # Allocate G1 (shift register)
  sr1 = np.ones(POLYNOMIAL_1.shape)

  # Allocate G2 (shift register)
  sr2 = np.ones(POLYNOMIAL_2.shape)

  # Get phase tap indices for selected PRN
  phase_taps = get_phase_taps(prn=prn)

  # Maximum-length shift register (MLSR)
  # Length is ((Length of 2^SR_SIZE) - 1) = 1023
  L = (2 ** SR_SIZE) - 1

  # Allocate G3 (output)
  G = np.zeros(L)

  # Generate C/A code sequence
  for i in range(L):
    # Take values from sr2 using the index specified by the PRN
    phase_tapped = np.mod(np.sum(sr2[phase_taps]), 2)

    # Calculates the current C/A code sequence
    G[i] = np.mod(sr1[-1] + phase_tapped, 2);

    # Calculates last value in shift register sr1
    # BEFORE rotating to the right
    sr1[-1] = np.mod(np.sum(sr1 * POLYNOMIAL_1), 2)
    # Once rotated, the last value is now the first value
    sr1 = np.roll(sr1, 1)

    # Do the same for shift register sr2
    sr2[-1] = np.mod(np.sum(sr2 * POLYNOMIAL_2), 2)
    sr2 = np.roll(sr2, 1)

  return G

if __name__ == '__main__':
  for prn in range(1, len(PHASE_TAPS) + 1):
    first_ten_chips = generate(prn=prn)[0:10]
    print 'PRN {0}: {1}'.format(prn, first_ten_chips)
