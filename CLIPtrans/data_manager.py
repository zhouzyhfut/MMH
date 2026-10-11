import numpy as np

from paths import feature_path, list_path, session_id


def process_test_regdb(mode='val', modal='visible'):
    """Load one evaluation list file.

    mode:  'val' (validation lists) or 'test' (test lists)
    modal: 'visible' (audio), 'thermal' (text) or 'depth' (video)

    Paths are read from the list file (relative to DATA_ROOT) and returned as
    absolute paths.
    """
    input_data_path = list_path(mode, modal)

    file_image = []
    file_label = []
    id_label = []
    with open(input_data_path, 'rt') as file:
        for line in file:
            parts = line.split()
            if len(parts) < 2:
                continue
            relative_path, label = parts[0], int(parts[1])
            file_image.append(feature_path(relative_path))
            file_label.append(label)
            id_label.append(session_id(relative_path))

    return file_image, np.array(file_label), np.array(id_label)
