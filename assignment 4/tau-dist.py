import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def main():
    # Load data
    price = np.loadtxt('Gold-dt=1min.txt', usecols=0)
    data_len = len(price)

    # price_dif = np.array([10, 20, 50, 100])
    price_dif = 100
    tau_list = []  # array of time interval between price going up
    taup_list = []  # array of time interval between price going down

    for i in range(0, data_len, 30):
        up_flag = True
        down_flag = True
        for t in range(i + 1, data_len):
            delta_P = price[t] - price[i]
            if up_flag and price_dif <= delta_P:
                tau_list.append(t - i)
                up_flag = False
            if down_flag and delta_P <= -price_dif:
                taup_list.append(t - i)
                down_flag = False
            if not up_flag and not down_flag:
                break

    tau_avg = np.mean(tau_list)
    taup_avg = np.mean(taup_list)
    print(f"tau_avg: {tau_avg}\ntau'_avg: {taup_avg}")

    plt.figure(figsize=(10, 6))
    sns.kdeplot(np.array(tau_list), label=f"$\\tau$ mean = {tau_avg:.1f}")
    sns.kdeplot(np.array(taup_list), label=f"$\\tau'$ mean = {taup_avg:.1f}")
    plt.title(f"KDE Plots of $\\tau$ and $\\tau'$ for $\\Delta P =${price_dif}")
    plt.xlabel('Time step')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"fig/DP{price_dif}.png")
    plt.show()


if __name__ == '__main__':
    main()
