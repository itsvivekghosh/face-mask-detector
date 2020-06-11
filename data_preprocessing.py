import cv2
import os
import sys
import re
import pandas as pd
import numpy as np


# Data Preprocessing
class Process:

    def __init__(self, categories, data_path):
        self.categories = categories
        self.data_path=data_path

    def find_labels(self):

        print("Labels are:")
        for i in self.categories:
            print(i, end=" and ")  # actual labels

        labels = [i for i in range(len(self.categories))]
        labels_dict = dict(zip(self.categories, labels))

        print("Labels are:", labels)
        print("Actual Labels:", labels_dict)

        return labels, labels_dict

    def pre_process_image(self, labels_dict, data_path):

        IMG_SIZE = 100
        data, target = list(), list()

        for category in self.categories:
            folder_path = os.path.join(data_path, category)
            img_names = os.listdir(folder_path)

            for img_name in img_names:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)

                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    resized_images = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                    data.append(resized_images)
                    target.append(labels_dict[category])

                except Exception as e:
                    print("ERROR: Error occurred while retrieving the file", e)
                    pass

        # print(len(data), len(target))
        # print(data[0], categories[target[0]])

        data = np.array(data) / 255.
        # print(data.shape)
        data = np.reshape(data, (data.shape[0], IMG_SIZE, IMG_SIZE, 1))
        # print(data.shape)

        from keras.utils.np_utils import to_categorical
        new_target = to_categorical(target)

        data_file, target_file = "data_", "target"
        print("Saving the data and Data as {} and {}".format(data_file, target_file))
        np.save(data_file, data)
        np.save(target_file, new_target)
        print("Saved as {} and {}".format(data_file, target_file))


def data_process_main():
    data_path = 'data/'
    categories = os.listdir(data_path)

    process = Process(categories, data_path)

    labels, labels_dict = process.find_labels()
    process.pre_process_image(labels_dict, data_path=data_path)


if __name__ == '__main__':
    data_process_main()