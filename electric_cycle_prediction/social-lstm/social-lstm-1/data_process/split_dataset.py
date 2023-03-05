import os
import pandas as pd


def split_data(csv_path, out_dir):
    # read csv
    df = pd.read_csv(csv_path, header=None)
    annos = df.values.T
    # select elec-cycle
    select_df = df.loc[:, df.loc[4, :] == 2]
    frames = sorted(list(set(list(select_df.loc[0, :]))))
    len_frams = len(frames)
    train_frame = frames[:int(len_frams * 0.6)]
    val_frame = frames[int(len_frams * 0.6):int(len_frams * 0.8)]
    test_frame = frames[int(len_frams * 0.8):]

    train_df = select_df.loc[:, select_df.loc[0, :] <= max(train_frame)]
    val_df = select_df.loc[:, select_df.loc[0, :] <= max(val_frame)]
    val_df = val_df.loc[:, val_df.loc[0, :] >= min(val_frame)]
    test_df = select_df.loc[:, select_df.loc[0, :] >= min(test_frame)]

    train_df.to_csv(out_dir + "train.csv", header=0, index=0)
    val_df.to_csv(out_dir + "validation.csv", header=0, index=0)
    test_df.to_csv(out_dir + "test.csv", header=0, index=0)


if __name__ == "__main__":
    # 改成自己输入的csv文件
    csv_path = "/media/huangluying/F/money/HNU.csv"
    # 改成自己希望的输出位置
    out_dir = "/media/huangluying/F/money/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    split_data(csv_path, out_dir)
