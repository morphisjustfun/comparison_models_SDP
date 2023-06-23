from preprocessing.sflds.main import run as sflds_preprocessing
from preprocessing.tree_ensembles.main import run as tree_ensembles_preprocessing

from models.sflds.main import run as sflds_run
from models.tree_ensembles.main import run as tree_ensembles_run


def preprocessing():
    versions = ['1.0', '1.2', '1.4', '1.6']
    sflds_preprocessing(versions=versions)
    tree_ensembles_preprocessing(versions=versions)


def run_models():
    sflds_run()
    tree_ensembles_run()


if __name__ == '__main__':
    preprocessing()
    run_models()
