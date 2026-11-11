import pytest
from PIL import Image
from preprocessing.image_budget import prepare_image,BUDGETS

def test_budget_cap_and_aspect(tmp_path):
    p=tmp_path/'x.png';Image.new('RGB',(2400,1800)).save(p)
    im=prepare_image(p,'low')
    assert im.metadata['width']*im.metadata['height']<=BUDGETS['low']
    assert abs(im.metadata['width']/im.metadata['height']-4/3)<0.01
    assert im.metadata['visual_tokens_actual'] is None

def test_adaptive_uses_question(tmp_path):
    p=tmp_path/'x.png';Image.new('RGB',(100,100)).save(p)
    assert prepare_image(p,'adaptive','读取文字').metadata['selected_budget']=='high'
    assert prepare_image(p,'adaptive','描述场景').metadata['selected_budget']=='low'
    assert prepare_image(p,'baseline').metadata['width']==100

def test_invalid_budget(tmp_path):
    with pytest.raises(ValueError):prepare_image(tmp_path/'x','unknown')
