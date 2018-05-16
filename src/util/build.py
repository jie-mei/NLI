import os
import glob

from typing import List, Dict, Union


BUILD_DIR = './build'


def parse_args(sys_argv: List[str]) -> Dict[str, Union[str, int, float, bool]]:
    """ Parse command line arguments to dictionary. The numeric values are auto
    convert to the according types. """
    kwargs = {}  # type: Dict[str, Union[str, int, float, bool]]
    if len(sys_argv) > 1:
        for arg in sys_argv[1:]:
            k = arg.split("=")[0][2:]
            v = arg.split("=")[1]  # type: Union[str, int, float, bool]
            if v == 'True':
                v = True
            elif v == 'False':
                v = False
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            kwargs[k] = v
    return kwargs


def get_model_path(model_name: str) -> str:
    return './build/models/{}'.format(model_name)


def get_new_default_model_path() -> str:
    """ Get a model name 'modeli' with the minimum positive i and do not exist
    in the build directory.
    """
    model_path = get_model_path('model')
    if not os.path.isdir(model_path):
        return model_path
    i = 1
    while True:
        if not os.path.isdir(model_path + str(i)):
            return model_path + str(i)
        i += 1


def get_last_default_model_path() -> str:
    """ Get a model path 'modeli' with the maximum i that exists in the build
    directory.
    """
    for p in reversed(sorted(glob.glob(get_model_path('*')))):
        if os.path.isdir(p):
            return p
    return get_model_path('model')


def get_log_path(model_path: str) -> str:
    """ Get the pathname to the log path.
    """
    return os.path.join(model_path, 'log')


def get_save_path(model_path: str) -> str:
    """ Get the pathname to the save file.
    """
    return os.path.join(model_path, 'model')


def get_saved_model(model_path: str, step: int = None) -> str:
    #print('{}/*.index'.format(model_path))
    #print(glob.glob('{}/*.index'.format(model_path)))
    saved_path = '{}/model-{}*.index'.format(model_path, step if step else '')
    # load the lastest trained model
    offset = len(model_path) + 7
    idx_file = sorted(glob.glob(saved_path),
            key=lambda name: int(name[offset:name.rfind('.')]))[-1]
    return os.path.splitext(idx_file)[0]
