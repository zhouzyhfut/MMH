import numpy as np
import pandas as pd
import torch.utils.data as data

from paths import feature_path, list_path, session_id


class CLIPdataset1(data.Dataset):
    def __init__(self, transform=None):
        # The data location is configured in paths.py (DATA_ROOT).
        video_file, video_id = load_data(list_path('trainvideo'))
        text_file, text_id = load_data(list_path('traintext'))

        self.train_color_image = video_file
        self.train_color_id = video_id
        self.traintext = text_file
        self.traintext_id = text_id
        self.transform = transform

    def __getitem__(self, index):
        traintext = self.traintext[index]
        with open(traintext, 'r', encoding='utf-8') as f:
            cliptext = f.read().strip()
        video1 = self.train_color_image[index]
        video1 = pd.read_csv(video1).values
        video1 = np.mean(video1, axis=0)
        if self.transform:
            video1 = self.transform(video1)
        return video1, cliptext

    def __len__(self):
        return len(self.train_color_id)


def load_data(input_data_path):
    """Read a list file and return (absolute paths, session ids)."""
    with open(input_data_path, 'rt') as file:
        data_file_list = file.read().splitlines()

    entries = [line.split() for line in data_file_list if line.strip()]
    file_image = [feature_path(entry[0]) for entry in entries]
    id_label = [session_id(entry[0]) for entry in entries]
    return file_image, id_label
