import pytest
from PIL import Image
from preprocessing.image_budget import BUDGETS, adaptive_budget_v2, prepare_image

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


@pytest.mark.parametrize(("width", "height", "question", "expected"), [
    (620, 836, "How many very strong democrats are there?", "low"),
    (800, 600, "Read the OCR text.", "medium"),
    (800, 600, "What is the difference between two bars?", "medium"),
    (800, 600, "How much more did people spend in 2013 than they did in 2011?", "medium"),
    (800, 600, "What is the combined total across both series?", "medium"),
    (800, 600, "比较两个系列的平均值。", "medium"),
    (800, 1796, "Find the state labelled 40%.", "high"),
    (250, 1800, "Find the state labelled 40%.", "low"),
    (1800, 800, "Compare these two bars.", "medium"),
])
def test_adaptive_v2_routing(width, height, question, expected):
    assert adaptive_budget_v2(width, height, question) == expected


def test_adaptive_versions_keep_their_own_routing(tmp_path):
    path = tmp_path / "chart.png"
    Image.new("RGB", (800, 600)).save(path)
    question = "How many very strong democrats are there?"
    assert prepare_image(path, "adaptive", question).metadata["selected_budget"] == "high"
    assert prepare_image(path, "adaptive_v2", question).metadata["selected_budget"] == "low"
