"""Single place that knows where the data live.

After cloning the repository only ``DATA_ROOT`` below has to be edited (or the
``MMH_DATA_ROOT`` environment variable exported).

Expected layout::

    <DATA_ROOT>/
    |-- trainpath/
    |   |-- trainaudiopath.txt              # /0/07011970/combined_mfcc.npy 0
    |   |-- traintextpath.txt               # /0/07011970/07011970.npy 0
    |   `-- trainvideopath.txt              # /0/07011970/video.csv 0
    |-- valpath/
    |   |-- valaudiopath.txt
    |   |-- valtextpath.txt
    |   `-- valvideopath.txt
    |-- testpath/
    |   |-- testaudiopath.txt
    |   |-- testtextpath.txt
    |   `-- testvideopath.txt
    `-- 0/07011970/                         # feature files
        |-- combined_mfcc.npy
        |-- 07011970.npy
        `-- video.csv

Every line of a list file holds a path *relative to <DATA_ROOT>* followed by its
label, e.g. ``/0/07011970/video.csv 0``.
"""

import os

# ---------------------------------------------------------------------------
# EDIT ME: root folder holding both the list files and the feature files.
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get('MMH_DATA_ROOT', 'YOUR_DATA_PATH')

# Sub-folder of DATA_ROOT holding the list files, per split.
_LIST_DIRS = {'train': 'trainpath', 'val': 'valpath', 'test': 'testpath'}

# Modality name used by the scripts -> prefix of the corresponding list file.
_MODAL_PREFIX = {'visible': 'audio', 'thermal': 'text', 'depth': 'video'}


def feature_path(relative_path):
    """Absolute path of a feature file referenced by a list file.

    >>> feature_path('/0/07011970/video.csv')
    'YOUR_DATA_PATH/0/07011970/video.csv'
    """
    return DATA_ROOT.rstrip('/\\') + '/' + relative_path.strip().replace('\\', '/').lstrip('/')


def list_path(split, modal):
    """Absolute path of a list file.

    >>> list_path('val', 'depth')
    'YOUR_DATA_PATH/valpath/valvideopath.txt'
    """
    return os.path.join(DATA_ROOT, _LIST_DIRS[split], f'{split}{_MODAL_PREFIX[modal]}path.txt')


def session_id(relative_path):
    """Session identifier of an entry: the folder right below the domain folder.

    >>> session_id('/0/07011970/video.csv')
    '07011970'
    >>> session_id('/3/T03211101301/Audio1/mfcc.npy')
    'T03211101301'
    """
    parts = relative_path.strip().replace('\\', '/').lstrip('/').split('/')
    if len(parts) < 2:
        raise ValueError(f'Cannot extract a session id from {relative_path!r}')
    return parts[1]
