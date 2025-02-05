import numpy as np

# Load data
gold = np.loadtxt('Gold-dt=1min.txt', usecols=0)
data_len = len(gold)

# Moving average
window_size = 11
gold_moving_avg = np.zeros(data_len - window_size + 1)
for i in range(data_len - window_size + 1):
    gold_moving_avg[i] = np.average(gold[i:i+window_size])

# save data
np.savetxt('gold_moving_avg.txt', gold_moving_avg)
