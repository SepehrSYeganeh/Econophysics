import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import itertools


def op_generator(dim):
    return list(itertools.permutations([i for i in range(dim)]))


def raw_data_to_ordinal_pattern_data(data):
    op_data = np.zeros(data_len - dimension + 1, dtype=np.int8)
    for i in range(data_len - dimension + 1):
        window = data[i:i + dimension]
        window_sorted = np.sort(window)
        op = tuple(np.where(window_sorted == datum)[0][0] for datum in window)
        for j in range(op_len):
            if op == ordinal_patterns[j]:
                op_data[i] = j
    return op_data


def delayed_data(x_raw, y_raw, delay=1):
    x_new = np.array(x_raw[:-delay])
    y_new = np.array(y_raw[:-delay])
    x_delayed = np.array(x_raw[delay:])
    return x_new, y_new, x_delayed


def probabilities(x_, y_, z_):
    n = len(x_)
    jp = np.zeros((op_len, op_len, op_len))
    mp_xy = np.zeros((op_len, op_len))
    mp_xz = np.zeros((op_len, op_len))
    p_x = np.zeros(op_len)
    for i in range(n):
        jp[x_[i]][y_[i]][z_[i]] += 1
        mp_xy[x_[i]][y_[i]] += 1
        mp_xz[x_[i]][z_[i]] += 1
        p_x[x_[i]] += 1
    return jp / n, mp_xy / n, mp_xz / n, p_x / n


def transfer_entropy1(jp, mp_xy, mp_xz, p_x):
    return H2(mp_xy) + H2(mp_xz) - H1(p_x) - H3(jp)


def H1(prob):
    s = 0
    for x in range(len(prob)):
        if prob[x] != 0:
            s -= prob[x] * np.log2(prob[x])
    return s


def H2(prob):
    s = 0
    for x in range(len(prob)):
        for y in range(len(prob[x])):
            if prob[x][y] != 0:
                s -= prob[x][y] * np.log2(prob[x][y])
    return s


def H3(prob):
    s = 0
    for x in range(len(prob)):
        for y in range(len(prob[x])):
            for z in range(len(prob[x][y])):
                if prob[x][y][z] != 0:
                    s -= prob[x][y][z] * np.log2(prob[x][y][z])
    return s


def transfer_entropy2(jp, mp_xy, mp_xz, p_x):
    te = 0
    for x in range(6):
        for y in range(6):
            for z in range(6):
                if mp_xy[x][y] and mp_xz[x][z] and p_x[x] and jp[x][y][z]:
                    te += jp[x][y][z]*np.log2(jp[x][y][z]*p_x[x]/mp_xy[x][y]/mp_xz[x][z])
    return te


def net_flow(T, x, y):
    return (abs(T[x][y]) - abs(T[y][x])) / (abs(T[x][y]) + abs(T[y][x]))


data_all = np.loadtxt("Ex3-test.txt", usecols=(1, 2, 3)).transpose()
data_len = len(data_all[0])  # number of datum in each data
time_series_num = len(data_all)  # number of time series

dimension = 3  # dimension of ordinal pattern
ordinal_patterns = op_generator(dimension)
op_len = len(ordinal_patterns)  # number of op
op_data_all = np.array([raw_data_to_ordinal_pattern_data(data) for data in data_all])
op_data_len = len(op_data_all[0])

transfer_entropy_matrix1 = np.zeros((time_series_num, time_series_num))
transfer_entropy_matrix2 = np.zeros((time_series_num, time_series_num))
for i in range(time_series_num):
    for j in range(time_series_num):
        x_hat, y_hat, z_hat = delayed_data(op_data_all[i], op_data_all[j])
        joint, marginal_xy, marginal_xz, prob_x = probabilities(x_hat, y_hat, z_hat)
        transfer_entropy_matrix1[i][j] = transfer_entropy1(joint, marginal_xy, marginal_xz, prob_x)
        transfer_entropy_matrix2[i][j] = transfer_entropy2(joint, marginal_xy, marginal_xz, prob_x)

net_entropy_flow1 = np.zeros((time_series_num, time_series_num))
net_entropy_flow2 = np.zeros((time_series_num, time_series_num))
for i in range(time_series_num):
    for j in range(time_series_num):
        if i != j:
            net_entropy_flow1[i][j] = net_flow(transfer_entropy_matrix1, i, j)
            net_entropy_flow2[i][j] = net_flow(transfer_entropy_matrix2, i, j)

te_heatmap1 = sn.heatmap(transfer_entropy_matrix1, annot=True, fmt=".4f", cmap="Reds", robust=True)
plt.title("Transfer Entropy 1", fontsize=20)
te_heatmap1.get_figure().savefig("fig/transfer-entropy1.png")
plt.show()
plt.close()

te_heatmap2 = sn.heatmap(transfer_entropy_matrix2, annot=True, fmt=".4f", cmap="Reds", robust=True)
plt.title("Transfer Entropy 2", fontsize=20)
te_heatmap2.get_figure().savefig("fig/transfer-entropy2.png")
plt.show()
plt.close()

nf_heatmap1 = sn.heatmap(net_entropy_flow1, center=0, annot=True, fmt=".3f", cmap="coolwarm", robust=True)
plt.title("Net Entropy Flow 1", fontsize=20)
nf_heatmap1.get_figure().savefig("fig/net_flow1.png")
plt.show()
plt.close()

nf_heatmap2 = sn.heatmap(net_entropy_flow2, center=0, annot=True, fmt=".3f", cmap="coolwarm", robust=True)
plt.title("Net Entropy Flow 2", fontsize=20)
nf_heatmap2.get_figure().savefig("fig/net_flow2.png")
plt.show()
plt.close()
