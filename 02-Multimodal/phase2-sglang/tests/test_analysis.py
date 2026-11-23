import pytest
from analysis.plot_results import percentile,analyze

def test_percentile():
    assert percentile([1,3],0.5)==2
    assert percentile([],0.9) is None

def test_no_fabricated_plots(tmp_path):
    source=tmp_path/'empty.jsonl';source.write_text('')
    with pytest.raises(ValueError):analyze(source,tmp_path/'plots')
    assert not (tmp_path/'plots').exists()
