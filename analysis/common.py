"""Parts shared by the analysis modules: folders, the per-module cache and the output-directory guard."""
import os
import pickle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO_ROOT, 'experiments', 'results')
DEFAULT_RESULTS_DIR = os.path.join(RESULTS_ROOT, 'cs_alp_steady_state')
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, 'experiments', 'figures')
CACHE_DIR = 'analysis_cache'


def cached(results_dir, file_name, version, build, refresh=False):
    """Return (data, 'cache' or 'built').

    data is read from <results_dir>/analysis_cache/<file_name> when that file exists, carries `version` and
    refresh is False; otherwise data = build() is computed, tagged with `version` and saved there.
    This is the only file an analysis module writes inside the results tree.
    """
    path = os.path.join(results_dir, CACHE_DIR, file_name)
    if not refresh and os.path.exists(path):
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        if data.get('version') == version:
            return data, 'cache'
    data = build()
    data['version'] = version
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data, 'built'


def check_out_dir(out_dir):
    """Resolve the output directory; raise ValueError if it lies in experiments/results."""
    results = os.path.realpath(RESULTS_ROOT)
    resolved = os.path.realpath(out_dir)
    if resolved == results or resolved.startswith(results + os.sep):
        raise ValueError('the output directory must not be inside experiments/results')
    return resolved
