import pytest
from evaluation.metrics import score,edit_distance

def test_metrics():
    assert score('The cat.',['cat'])==1
    assert score('cat',['cat']*3+['dog']*7,'vqa')==pytest.approx(0.9)
    assert score('abcd',['abce'],'anls')==0.75
    assert score('ab',['ac'],'anls')==0.0
    assert edit_distance('kitten','sitting')==3
    with pytest.raises(ValueError):score('a',[])
