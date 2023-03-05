import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    test_label = "/media/huangluying/F/money/20230202-sociallstm/social-lstm-1/data_process/test_label.txt"
    test = "/media/huangluying/F/money/20230202-sociallstm/social-lstm-1/result/SOCIALLSTM/LSTM/elec/test.txt"
    np_label = np.genfromtxt(test_label)
    np_test = np.genfromtxt(test)
    total_len = np_test.shape[0]

    for i in range(0, total_len, 20):
        cur_label = np_label[i:i+20, :]
        cur_test = np_test[i:i+20, :]
        plt.figure()
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.plot(cur_label[:, 3], cur_label[:, 2], c='r', marker='o')
        plt.plot(cur_test[:, 3], cur_test[:, 2], c='b', marker='.')
        ax = plt.gca()  # 获取到当前坐标轴信息
        ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
        ax.invert_yaxis()  # 反转Y坐
        plt.savefig(os.path.join("plot_test", str(cur_label[0, 1]) + '.jpg'))
        plt.close()
