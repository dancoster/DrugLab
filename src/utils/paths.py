import json
import collections.abc


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


overwrite_json = ""

with open('basic_config.json', 'r') as f:
    config = json.load(f)
if overwrite_json is not "":
    with open(overwrite_json, 'r') as f:
        overwrite_config = json.load(f)
        config = update(config, overwrite_config)
