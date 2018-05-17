from typing import List, Dict, Union

import yaml


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


def parse_yaml(path, mode=None):
    def flatten(prefix, to_flat):
        """ Flatten the nested value by joining their keys. """
        if not (isinstance(to_flat, dict) or
                isinstance(to_flat, list)):
            return {prefix: to_flat}
        flat = {}
        if isinstance(to_flat, dict):
            for k, v in to_flat.items():
                flat_k = prefix + '_' + k if prefix else k
                flat = {**flat, **flatten(flat_k, v)}
        else:
            for e in to_flat:
                flat = {**flat, **flatten(prefix, e)}
        return flat
    with open(path, 'r') as conf_file:
        raw = yaml.load(conf_file)
        conf = {**flatten(None, raw['model']),
                **flatten('data', raw['data'])}
        if mode and mode in raw:
                conf = {**conf, **flatten(None, raw[mode])}
        return conf
