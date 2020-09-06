    # from __future__ import print_function
from __future__ import division
import numpy as np
import sys
import random
import os
import shutil
import pickle 


def main():
    # Global root of MD dataset
    root = "/home/konstantin/datasets/MegaDepthDataset/"
    # Root of v1 directory
    parent_folder = "MegaDepth_v1"
    # orientation_list = ["landscape", "portrait"]
    orientation_list = ["landscape"]

    ratio = 0

    for orientation in orientation_list:

        dir_load_all_img = root +"/train_val_list/" + orientation + "/imgs_MD.p"
        # dir_load_all_target = root + "/train_val_list/" + orientation + "/targets_MD.p"

        dir_save_train_img = root + "/train_list/" + orientation + "/imgs_MD.p"
        # dir_save_train_target = root + "/train_list/" + orientation + "/targets_MD.p"

        # dir_save_val_img = root + "final_list/val_list/" + orientation + "/imgs_MD.p"
        # dir_save_val_target = root + "final_list/val_list/" + orientation + "/targets_MD.p"


        img_list = np.load(dir_load_all_img, allow_pickle=True)
        # target_list = np.load(dir_load_all_target, allow_pickle=True)

        val_num = int(round(ratio * len(img_list)) )

        # shuffle_list = (np.arange(len(img_list)))
        # random.shuffle(shuffle_list)

        train_img_list = []
        # train_targets_list =[]
        val_img_list = []
        # val_targets_list =[]

        for i in range(0, val_num):
            # val_targets_list.append(target_list[shuffle_list[i]])
            # val_img_list.append(img_list[shuffle_list[i]])
            val_img_list.append(os.path.join(parent_folder, img_list[i]))

        for i in range(val_num, len(img_list)):
            # train_targets_list.append(target_list[shuffle_list[i]])
            # train_img_list.append(img_list[shuffle_list[i]])
            train_img_list.append(os.path.join(parent_folder, img_list[i]))

        print("orientation: %s"%orientation)
        print("train list length : %d"%(len(train_img_list)))
        print("validation list length : %d"%(len(val_img_list)))


        # save train list
        img_list_file = open(dir_save_train_img, 'wb')
        pickle.dump(train_img_list, img_list_file)
        img_list_file.close()

        # img_list_file = open(dir_save_train_target, 'wb')
        # pickle.dump(train_targets_list, img_list_file)
        # img_list_file.close()


        # save validation list
        # img_list_file = open(dir_save_val_img, 'wb')
        # pickle.dump(val_img_list, img_list_file)
        # img_list_file.close()

        # img_list_file = open(dir_save_val_target, 'wb')
        # pickle.dump(val_targets_list, img_list_file)
        # img_list_file.close()



if __name__ == "__main__":
    main()

# 
