''' Configuration File.
'''

CUDA_VISIBLE_DEVICES = 0
NUM_TRAIN = 50000 # N \Fashion MNIST 60000, cifar 10/100 50000
BATCH     = 128 # B

MARGIN = 1.0 # xi
WEIGHT = 1.0 # lambda

TRIALS = 3
CYCLES = 10

# cifar
EPOCH = 200
EPOCH_GCN = 200
LR = 1e-1
LR_GCN = 1e-3
MILESTONES = [160, 260]
EPOCHL = 120  #20 #120 # After 120 epochs, stop
EPOCHV = 100  # VAAL number of epochs


MOMENTUM = 0.9
WDECAY =5e-4  #2e-3# 5e-4

