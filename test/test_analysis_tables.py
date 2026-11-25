import os
import pickle
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import common  # noqa: E402
from analysis.tables import case_study_thresholds as t6  # noqa: E402
from analysis.tables import saure_ejor_steady_state as t5  # noqa: E402


# ---------- common: cache and output guard ----------

def test_cache_is_built_once_reused_and_rebuilt(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return {'value': len(calls)}

    data, source = common.cached(str(tmp_path), 'thing.pkl', 3, build)
    assert (source, data['value'], data['version']) == ('built', 1, 3)
    assert os.listdir(tmp_path) == ['analysis_cache']
    assert os.listdir(tmp_path / 'analysis_cache') == ['thing.pkl']
    data, source = common.cached(str(tmp_path), 'thing.pkl', 3, build)
    assert (source, data['value'], len(calls)) == ('cache', 1, 1)
    data, source = common.cached(str(tmp_path), 'thing.pkl', 3, build, refresh=True)
    assert (source, data['value']) == ('built', 2)
    data, source = common.cached(str(tmp_path), 'thing.pkl', 4, build)
    assert (source, data['value'], data['version']) == ('built', 3, 4)


def test_output_guard_refuses_the_results_tree(tmp_path):
    with pytest.raises(ValueError):
        common.check_out_dir(os.path.join(common.RESULTS_ROOT, 'anything'))
    with pytest.raises(ValueError):
        common.check_out_dir(common.RESULTS_ROOT)
    assert common.check_out_dir(str(tmp_path)) == os.path.realpath(str(tmp_path))


# ---------- Table 6 ----------

def _t6_numbers(day20=(100.0, 0.0)):
    days = [1, 5, 10, 15, 20]
    def entry(base):
        return {d: (day20 if d == 20 else (base + d, 0.25 + d / 100)) for d in days}
    types = {1: {'PH': entry(82.6), 'ALP': entry(63.4), 'M': entry(55.6)},
             2: {'PH': entry(7.0), 'ALP': entry(2.04), 'M': entry(3.0)}}
    total = {'PH': entry(18.9), 'ALP': entry(22.5), 'M': entry(4.6)}
    return {'version': t6.CACHE_VERSION, 'days': days, 'policies': ['PH', 'ALP', 'M'], 'n_paths': 2048,
            'confidence': 0.95, 'types': types, 'total': total}


def test_table6_cells_and_rows():
    assert t6.cell((83.2, 1.06)) == '\\(83{\\pm}1.1\\)'
    assert t6.cell((19.4, 0.46), bold=True) == '\\(\\mathbf{19{\\pm}0.5}\\)'
    lines = t6.latex(_t6_numbers()).split('\n')
    # PH has the highest mean in every column of type 1; the 20-workday entry is one merged cell
    assert ('1 & \\(\\mathbf{84{\\pm}0.3}\\) & \\(64{\\pm}0.3\\) & \\(57{\\pm}0.3\\) & '
            '\\(\\mathbf{88{\\pm}0.3}\\) & \\(68{\\pm}0.3\\) & \\(61{\\pm}0.3\\) & '
            '\\(\\mathbf{93{\\pm}0.3}\\) & \\(73{\\pm}0.3\\) & \\(66{\\pm}0.3\\) & '
            '\\(\\mathbf{98{\\pm}0.4}\\) & \\(78{\\pm}0.4\\) & \\(71{\\pm}0.4\\) & \\(100{\\pm}0.0\\) \\\\') in lines
    # the Total row uses the same rule as the type rows (ALP has the highest means here) and its label is not bold
    i = lines.index('Total')
    assert lines[i + 1:i + 6] == [
        '& \\(20{\\pm}0.3\\) & \\(\\mathbf{24{\\pm}0.3}\\) & \\(6{\\pm}0.3\\)',
        '& \\(24{\\pm}0.3\\) & \\(\\mathbf{28{\\pm}0.3}\\) & \\(10{\\pm}0.3\\)',
        '& \\(29{\\pm}0.3\\) & \\(\\mathbf{32{\\pm}0.3}\\) & \\(15{\\pm}0.3\\)',
        '& \\(34{\\pm}0.4\\) & \\(\\mathbf{38{\\pm}0.4}\\) & \\(20{\\pm}0.4\\)',
        '& \\(100{\\pm}0.0\\) \\\\',
    ]
    assert not any(line.startswith('\\textbf{Total}') for line in lines)


def test_table6_bold_follows_the_unrounded_means():
    numbers = _t6_numbers()
    numbers['types'][2]['PH'][1] = (2.04, 0.5)      # equal unrounded means at the top: both bold
    numbers['types'][2]['ALP'][1] = (2.04, 0.3)
    numbers['types'][2]['M'][1] = (1.0, 0.3)
    numbers['types'][2]['M'][5] = (12.0, 0.5)       # highest mean at 5 workdays
    numbers['types'][2]['PH'][5] = (11.6, 0.5)      # also shown as 12, but lower before rounding: not bold
    row = next(line for line in t6.latex(numbers).split('\n') if line.startswith('2 & '))
    cells = row.split(' & ')[1:]
    assert cells[0] == '\\(\\mathbf{2{\\pm}0.5}\\)' and cells[1] == '\\(\\mathbf{2{\\pm}0.3}\\)'
    assert cells[3] == '\\(12{\\pm}0.5\\)' and cells[5] == '\\(\\mathbf{12{\\pm}0.5}\\)'


def test_table6_no_bold_when_all_three_means_are_equal():
    numbers = _t6_numbers()
    for name in ('PH', 'ALP', 'M'):
        numbers['types'][1][name][15] = (100.0, 0.0)
    row = next(line for line in t6.latex(numbers).split('\n') if line.startswith('1 & '))
    assert row.split(' & ')[10:13] == ['\\(100{\\pm}0.0\\)'] * 3


def test_table6_header_and_note():
    text = t6.latex(_t6_numbers())
    assert '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l*{13}{c}@{}}' in text
    assert ('& \\multicolumn{3}{c}{1 workday}\n& \\multicolumn{3}{c}{5 workdays}\n'
            '& \\multicolumn{3}{c}{10 workdays}\n& \\multicolumn{3}{c}{15 workdays}\n& 20 workdays \\\\') in text
    assert '\\cmidrule(lr){11-13}\n\\cmidrule(lr){14-14}' in text
    assert '& PH & ALP & M & PH & ALP & M & PH & ALP & M & PH & ALP & M & PH, ALP, M \\\\' in text
    assert ('\\item \\textit{Note.} Entries are percentages reported as sample means \\(\\pm\\) 95\\% '
            'confidence-interval half-widths over 2,048 evaluation sample paths. For the 20-workday threshold, the '
            'entries of the three policies are equal and are shown once. Bold entries are the largest among the three '
            'policies for the same type and threshold, compared before rounding. PH, ALP, and M denote') in text


def test_table6_refuses_to_merge_20_workday_entries_that_differ():
    numbers = _t6_numbers()
    numbers['types'][2]['M'][20] = (99.4, 0.0)        # shown as 99, the other two as 100
    with pytest.raises(ValueError, match='20-workday'):
        t6.latex(numbers)
    numbers = _t6_numbers()
    numbers['total']['PH'][20] = (100.0, 0.06)        # shown as 0.1, the other two as 0.0
    with pytest.raises(ValueError, match='20-workday'):
        t6.latex(numbers)
    numbers = _t6_numbers(day20=(98.0, 0.5))          # equal as displayed in every row: allowed
    assert '& \\(98{\\pm}0.5\\) \\\\' in t6.latex(numbers)


# ---------- Table 5 ----------

def _t5_numbers():
    cells = {
        'approx_penalized_hindsight': {'cost': (68815.2, 2737.21), 'improvement': (46.31, 0.84), 'violation': (10.2, 0.71),
                                       'overtime': (40.1, 0.24), 'zero_gap': (61.43, 2.41), 'penalized_gap': (12.12, 0.68),
                                       'partial_gap': (8.94, 3.21)},
        'row_gen_alp': {'cost': (99119.4, 5029.04), 'improvement': (22.7, 1.6), 'violation': (17.1, 1.0),
                        'overtime': (43.2, 0.2), 'zero_gap': (132.4, 7.9), 'penalized_gap': (62.4, 4.7),
                        'partial_gap': (56.9, 6.1)},
        'myopic': {'cost': (128202.1, 5239.5), 'improvement': None, 'violation': (45.0, 0.9),
                   'overtime': (31.2, 0.1), 'zero_gap': (200.6, 8.7), 'penalized_gap': (110.7, 4.9),
                   'partial_gap': (102.9, 7.0)},
    }
    return {'version': t5.CACHE_VERSION, 'n_paths': 2048, 'confidence': 0.95, 'cells': cells}


def test_table5_cells():
    assert t5.cell((68815.2, 2737.21), 0, bold=True) == '\\(\\mathbf{68{,}815 \\pm 2{,}737.2}\\)'
    assert t5.cell((132.4, 7.9), 1) == '\\(132.4 \\pm 7.9\\)'


def test_table5_rows_with_bold_from_the_data():
    lines = t5.latex(_t5_numbers()).split('\n')
    assert ('ALP-based PH & \\(\\mathbf{68{,}815 \\pm 2{,}737.2}\\) & \\(\\mathbf{46.3 \\pm 0.8}\\) & '
            '\\(\\mathbf{10 \\pm 0.7}\\) & \\(40 \\pm 0.2\\) & \\(\\mathbf{61.4 \\pm 2.4}\\) & '
            '\\(\\mathbf{12.1 \\pm 0.7}\\) & \\(\\mathbf{8.9 \\pm 3.2}\\) \\\\') in lines
    assert ('Myopic & \\(128{,}202 \\pm 5{,}239.5\\) & --- & \\(45 \\pm 0.9\\) & \\(\\mathbf{31 \\pm 0.1}\\) & '
            '\\(200.6 \\pm 8.7\\) & \\(110.7 \\pm 4.9\\) & \\(102.9 \\pm 7.0\\) \\\\') in lines


def test_table5_improvement_bold_goes_to_the_larger_value():
    numbers = _t5_numbers()
    numbers['cells']['row_gen_alp']['improvement'] = (50.0, 1.0)
    lines = t5.latex(numbers).split('\n')
    alp = next(line for line in lines if line.startswith('ALP &'))
    ph = next(line for line in lines if line.startswith('ALP-based PH &'))
    assert '\\(\\mathbf{50.0 \\pm 1.0}\\)' in alp
    assert '\\(46.3 \\pm 0.8\\)' in ph


def test_table5_header_and_note():
    text = t5.latex(_t5_numbers())
    assert '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} lccccccc @{}}' in text
    assert 'Policy\n& \\makecell{Discounted\\\\cost}\n& \\makecell{Improvement\\\\over Myopic (\\%)}' in text
    assert '& \\makecell{\\% gap to partial\\\\relaxation LB}\\\\\n\\midrule' in text
    assert '\\item \\textit{Note.} Bold entries are the best in each column.' in text


# ---------- real data (results are not in git) ----------

def _current(results_dir, name, version):
    path = os.path.join(results_dir, common.CACHE_DIR, name)
    if not os.path.exists(path):
        return False
    with open(path, 'rb') as handle:
        return pickle.load(handle).get('version') == version


@pytest.mark.skipif(not _current(common.DEFAULT_RESULTS_DIR, t6.CACHE_FILE, t6.CACHE_VERSION),
                    reason='results are not in git; needs a current cache')
def test_table6_total_row_with_the_real_results():
    numbers, source = t6.load_numbers(common.DEFAULT_RESULTS_DIR)
    assert source == 'cache'
    lines = t6.latex(numbers).split('\n')
    i = lines.index('Total')
    assert lines[i + 1:i + 6] == [
        '& \\(19{\\pm}0.5\\) & \\(\\mathbf{23{\\pm}0.9}\\) & \\(5{\\pm}0.3\\)',
        '& \\(\\mathbf{61{\\pm}0.6}\\) & \\(51{\\pm}0.9\\) & \\(36{\\pm}1.0\\)',
        '& \\(\\mathbf{95{\\pm}0.4}\\) & \\(94{\\pm}0.7\\) & \\(81{\\pm}0.9\\)',
        '& \\(98{\\pm}0.2\\) & \\(\\mathbf{100{\\pm}0.2}\\) & \\(99{\\pm}0.2\\)',
        '& \\(100{\\pm}0.0\\) \\\\',
    ]


@pytest.mark.skipif(not _current(common.DEFAULT_RESULTS_DIR, t5.CACHE_FILE, t5.CACHE_VERSION),
                    reason='results are not in git; needs a current cache')
def test_table5_discounted_costs_with_the_real_results():
    numbers, source = t5.load_numbers(common.DEFAULT_RESULTS_DIR)
    assert source == 'cache'
    costs = {pid: t5.cell(numbers['cells'][pid]['cost'], 0) for pid in numbers['cells']}
    assert costs == {'approx_penalized_hindsight': '\\(68{,}815 \\pm 2{,}737.2\\)',
                     'row_gen_alp': '\\(99{,}119 \\pm 5{,}029.0\\)',
                     'myopic': '\\(128{,}202 \\pm 5{,}239.5\\)'}
