"""Single place that knows where the pretraining data live.

After cloning the repository only ``DATA_ROOT`` below has to be edited (or the
``MMH_PRETRAIN_DATA_ROOT`` environment variable exported).

Expected layout::

    <DATA_ROOT>/
    |-- pretrainpath/
    |   |-- trainvideo.txt                  # /0/07011970/video_feature/video_01/video_01.csv
    |   `-- traintext.txt                   # /0/07011970/video_feature/video_01/video_01.txt
    `-- 0/07011970/video_feature/video_01/
        |-- video_01.csv                    # video features (one row per frame)
        `-- video_01.txt                    # caption/description read as CLIP text

Every line of a list file holds a path *relative to <DATA_ROOT>*, e.g.
``/0/07011970/video_feature/video_01/video_01.csv``.
"""

import os

# ---------------------------------------------------------------------------
# EDIT ME: root folder holding both the list files and the feature files.
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get('MMH_PRETRAIN_DATA_ROOT', 'YOUR_DATA_PATH')

# Sub-folder of DATA_ROOT holding the list files.
_LIST_DIR = 'pretrainpath'


def feature_path(relative_path):
    """Absolute path of a feature file referenced by a list file.

    >>> feature_path('/0/07011970/video_feature/video_01/video_01.csv')
    'YOUR_DATA_PATH/0/07011970/video_feature/video_01/video_01.csv'
    """
    return DATA_ROOT.rstrip('/\\') + '/' + relative_path.strip().replace('\\', '/').lstrip('/')


def list_path(name):
    """Absolute path of a list file.

    >>> list_path('trainvideo')
    'YOUR_DATA_PATH/pretrainpath/trainvideo.txt'
    """
    return os.path.join(DATA_ROOT, _LIST_DIR, f'{name}.txt')


def session_id(relative_path):
    """Session identifier of an entry: the folder right below the domain folder.

    >>> session_id('/0/07011970/video_feature/video_01/video_01.csv')
    '07011970'
    >>> session_id('/3/T03211101301/Video1/T03211101301_11-1/T03211101301_11-1.csv')
    'T03211101301'
    """
    parts = relative_path.strip().replace('\\', '/').lstrip('/').split('/')
    if len(parts) < 2:
        raise ValueError(f'Cannot extract a session id from {relative_path!r}')
    return parts[1]
