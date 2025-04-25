import os
import pickle


def load_pickle(path):
    if os.path.isfile(path) and os.path.splitext(path)[-1] == '.pkl':
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    else:
        return None


def save_pickle(path, obj):
    d = os.path.split(path)[0]
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
