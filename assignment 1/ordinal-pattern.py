import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import seaborn as sn


def count_ordinal_pattern(ord_pat: tuple, op_index):
    i = 0
    while i < o_p_len:
        if ord_pat == ordinal_patterns[i]:
            o_p_count[i] += 1
            o_p_arr[op_index] = i
            conditional_count_matrix[i][int(o_p_arr[op_index - 1])] += 1
            break
        i += 1


def turn_data_to_ordinal_pattern(window: list):
    window_sorted = np.sort(window)
    return tuple(np.where(window_sorted == datum)[0][0] for datum in window)


def detect_ordinal_pattern(window, op_index):
    count_ordinal_pattern(turn_data_to_ordinal_pattern(window), op_index)


data = np.loadtxt("gold_moving_avg.txt")
data_len = len(data)
dimension = 5
o_p_len = math.factorial(dimension)
ordinal_patterns = list(itertools.permutations([i for i in range(dimension)]))
o_p_arr = np.zeros(data_len - dimension + 1)
o_p_count = np.zeros(o_p_len)
conditional_count_matrix = np.zeros((o_p_len, o_p_len))  # [i][j]: i happened right after j

if __name__ == '__main__':
    for i in range(data_len - dimension + 1):
        detect_ordinal_pattern(data[i:i + dimension], i)

    conditional_probability_matrix = np.zeros((o_p_len, o_p_len))
    for i in range(o_p_len):
        for j in range(o_p_len):
            if o_p_count[j] != 0:
                conditional_probability_matrix[i][j] = conditional_count_matrix[i][j] / o_p_count[j]

    heatmap = sn.heatmap(data=conditional_probability_matrix)
    plt.show()
