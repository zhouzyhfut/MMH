import numpy as np
import pandas as pd
import torch.utils.data as data

from paths import feature_path, list_path, session_id


class RegDBData(data.Dataset):
    def __init__(self, transform=None):
        # The data location is configured in paths.py (DATA_ROOT).
        train_color_list = list_path('train', 'visible')
        train_thermal_list = list_path('train', 'thermal')
        train_depth_list = list_path('train', 'depth')

        color_img_file, train_color_label, color_id = load_data(train_color_list)
        thermal_img_file, train_thermal_label, thermal_id = load_data(train_thermal_list)
        depth_img_file, train_depth_label, depth_id = load_data(train_depth_list)

        self.train_color_image = color_img_file
        self.train_color_label = train_color_label
        self.train_color_id = color_id
        self.train_thermal_image = thermal_img_file
        self.train_thermal_label = train_thermal_label
        self.train_thermal_id = thermal_id
        self.train_depth_image = depth_img_file
        self.train_depth_label = train_depth_label
        self.train_depth_id = depth_id
        self.transform = transform

    def __getitem__(self, index):
        audio_path = self.train_color_image[index]
        target1 = self.train_color_label[index]
        id1 = self.train_color_id[index]
        text_path = self.train_thermal_image[index]
        target2 = self.train_thermal_label[index]
        id2 = self.train_thermal_id[index]
        video_path = self.train_depth_image[index]
        target3 = self.train_depth_label[index]
        id3 = self.train_depth_id[index]

        audio = np.load(audio_path)
        if target1 == 3:
            audio = audio.T
        text = np.load(text_path)
        video = pd.read_csv(video_path).values
        audio = np.mean(audio, axis=0)
        text = np.mean(text, axis=0)
        video = np.mean(video, axis=0)

        if self.transform:
            audio = self.transform(audio)
            text = self.transform(text)
            video = self.transform(video)

        return audio, text, video, target1, target2, target3, id1, id2, id3

    def __len__(self):
        return len(self.train_color_id)


class TestData(data.Dataset):
    def __init__(self, file_list, labels, ids, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __getitem__(self, index):
        path = self.file_list[index]
        label = self.labels[index]
        identity = self.ids[index]

        if path.endswith('.npy'):
            feat = np.load(path)
            if path.endswith('mfcc.npy') and label == 3:
                feat = feat.T
        elif path.endswith('.csv'):
            feat = pd.read_csv(path).values
        else:
            raise ValueError(f'Unsupported file type: {path}')

        feat = np.mean(feat, axis=0)
        if self.transform:
            feat = self.transform(feat)

        return feat, label, identity

    def __len__(self):
        return len(self.ids)


def load_data(input_data_path):
    """Read a list file and return (absolute feature paths, labels, session ids)."""
    with open(input_data_path, 'rt') as file:
        data_file_list = file.read().splitlines()

    entries = [line.split() for line in data_file_list if line.strip()]
    file_image = [feature_path(entry[0]) for entry in entries]
    file_label = [int(entry[1]) for entry in entries]
    id_label = [session_id(entry[0]) for entry in entries]
    return file_image, file_label, id_label
