import os
import cv2
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


class DataProcess:
    def __init__(self,
                 train_path,
                 val_path,
                 test_path,
                 txt_path,
                 out_path,
                 matrix_pair,
                 select_inds=2,
                 plot=False):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.txt_path = txt_path
        self.out_path = out_path
        img_coors, world_coor = matrix_pair
        self.select_inds = select_inds  # 选取的类别
        self.plot = plot  # 是否可视化轨迹

        # matrix, mask = cv2.findHomography(img_coors, world_coor, cv2.RANSAC, 5.0)
        self.affine = cv2.getPerspectiveTransform(world_coor, img_coors)

        # # check affine
        # img_points = np.array([[656, 494, 1], [836, 485, 1], [695, 961, 1], [1014, 952, 1]]).reshape(-1, 3).T
        # global_points = np.dot(np.linalg.inv(self.affine), img_points).T
        # global_points = global_points / global_points[:, 2].T.reshape(4, 1)

        x, y = self.cvt_pos([656, 494], np.linalg.inv(self.affine))
        print(x, y)
        x, y = self.cvt_pos([836, 485], np.linalg.inv(self.affine))
        print(x, y)
        x, y = self.cvt_pos([695, 961], np.linalg.inv(self.affine))
        print(x, y)
        x, y = self.cvt_pos([1014, 952], np.linalg.inv(self.affine))
        print(x, y)
        x, y = self.cvt_pos([1920, 1080], np.linalg.inv(self.affine))
        print(x, y)
        x, y = self.cvt_pos([1920, 0], np.linalg.inv(self.affine))
        print(x, y)
        x, y = self.cvt_pos([0, 1080], np.linalg.inv(self.affine))
        print(x, y)
        x, y = self.cvt_pos([0, 0], np.linalg.inv(self.affine))
        print(x, y)

    # 透视坐标转换
    def cvt_pos(self, pos, cvt_mat_t):
        u = pos[0]
        v = pos[1]
        x = (cvt_mat_t[0][0] * u + cvt_mat_t[0][1] * v + cvt_mat_t[0][2]) / (
                cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
        y = (cvt_mat_t[1][0] * u + cvt_mat_t[1][1] * v + cvt_mat_t[1][2]) / (
                cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
        return x, y

    def plot_traj(self, df):
        save_path = "./traj"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        def randomcolor():
            colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
            color = ""
            for i in range(6):
                color += colorArr[random.randint(0, 14)]
            return "#" + color

        select_df = df.loc[:, df.loc[4, :] == 2]
        frames = set(list(select_df.loc[0, :]))
        ids = set(list(select_df.loc[1, :]))
        # generate color
        color_dict = dict()
        for id in ids:
            color_dict[int(id)] = randomcolor()
        # 遍历帧
        for f in sorted(list(frames)):
            plt.figure()
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            cur_frame = select_df.loc[:, select_df.loc[0, :] == f].values
            for idx in range(cur_frame.shape[1]):
                y = cur_frame[2, idx]
                x = cur_frame[3, idx]
                cur_id = int(cur_frame[1, idx])
                x_world, y_world = self.cvt_pos([x, y], np.linalg.inv(self.affine))
                x_world /= 100
                y_world /= 100
                plt.scatter(x_world, y_world, c=color_dict[cur_id], s=15)
                plt.text(x_world, y_world, str(cur_id))
            ax = plt.gca()  # 获取到当前坐标轴信息
            ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
            ax.invert_yaxis()  # 反转Y坐标轴
            plt.savefig(os.path.join(save_path, str(int(f)) + '.jpg'))
            plt.close()

    def process(self):
        self.process_single(self.train_path, "train", "train")
        self.process_single(self.val_path, "validation", "validation")
        self.process_single(self.test_path, "validation", "test")

    def process_single(self, path, phase, out_name):
        # read csv
        df = pd.read_csv(path, header=None)
        if self.plot:
            self.plot_traj(df)
        annos = df.values.T
        # select elec-cycle
        select = annos[:, -1] == self.select_inds
        annos_keep = annos[select]
        # calculate elec-cycle ids
        select_ids = set(annos_keep[:, 1].astype(int).tolist())
        # process txt
        txt_files = os.listdir(self.txt_path)
        # filter ele-cycle
        select_fs = [f for f in txt_files if int(f.split('.txt')[0]) in select_ids]

        seq_cnt = 0
        seqs = []
        # for loop process txt files
        for sf in select_fs:
            cur_data = self.read_txt(os.path.join(self.txt_path, sf), float(sf.split('.txt')[0]))
            # split data
            if cur_data.shape[0] > 0:
                sp_data = self.split_data(cur_data)
                num_data = len(sp_data)
                seq_cnt += num_data
                seqs += sp_data
        print("seq num:", seq_cnt)
        np_seq_data = np.array(seqs)

        self.save_annos(np_seq_data, phase, out_name)

    def read_txt(self, txt_file, anno_id, anno_length=3):
        # anno length means the class anno include how many rows
        datas = []
        with open(os.path.join(self.txt_path, txt_file), 'r') as f:
            lines = f.readlines()
            total_infos = (len(lines) - 1) // 3
            # if < 4s continue
            if total_infos < 20:
                return np.zeros((0, 4))
            for i in range(total_infos):
                frame = float(lines[i * 3].strip())
                x = float(lines[i * 3 + 1].strip())  # u
                y = float(lines[i * 3 + 2].strip())  # v
                x_world, y_world = self.cvt_pos([x, y], np.linalg.inv(self.affine))
                cur_frame = [frame, anno_id, y_world / 100, x_world / 100]
                # cur_frame = [frame, anno_id, y, x]
                datas.append(cur_frame)
        datas = np.array(datas)
        return datas

    def split_data(self, arr, seq_length=20):
        # seq length means how much frame you want to use
        split_data = []
        data_length = arr.shape[0]
        split_num = data_length - seq_length + 1
        # frames = np.arange(seq_length)
        for i in range(split_num):
            cur_split = arr[i:i + seq_length, :]
            split_data.append(cur_split)
        return split_data

    def save_annos(self, arrs, phase, out_name):
        out_path = os.path.join(self.out_path, phase, 'elec')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_str = ""
        for arr in arrs:
            for idx, line in enumerate(arr):
                line_list = line.tolist()
                out_str += ' '.join([str(x) for x in line_list])
                out_str += '\n'
        out_str.strip()
        with open(os.path.join(out_path, out_name + '.txt'), 'w') as f:
            f.write(out_str)


if __name__ == "__main__":
    # 填入生成的train.csv文件
    train_path = "/media/huangluying/F/money/train.csv"
    # 填入生成的validation.csv文件
    val_path = "/media/huangluying/F/money/validation.csv"
    # 填入生成的test.csv文件
    test_path = "/media/huangluying/F/money/test.csv"
    # 填入txt标注的文件夹
    txt_path = "/media/huangluying/F/money/electric_cycle"
    # 填入输出data位置，这里一定要填入项目根目录的data文件夹
    data_out_path = "/media/huangluying/F/money/20230202-sociallstm/social-lstm-1/data"
    if not os.path.exists(data_out_path):
        os.makedirs(data_out_path)
    img_coors = np.float32([[656, 494], [836, 485], [695, 961], [1014, 952]])
    world_coors = np.float32([[3000, 4000], [3428, 4000], [3203.85, 5432.57], [3763.52, 5451.73]])
    process = DataProcess(train_path, val_path, test_path, txt_path, data_out_path, (img_coors, world_coors))

    process.process()
