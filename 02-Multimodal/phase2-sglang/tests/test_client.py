import json
import httpx
import pytest
from api.client import VLMClient,user_message

def transport(truncated=False):
    def handler(request):
        payload=json.loads(request.content)
        assert payload['stream_options']['include_usage'] is True
        events=[{'choices':[{'delta':{'role':'assistant'},'finish_reason':None}]},
                {'choices':[{'delta':{'content':'这是图片'},'finish_reason':None}]},
                {'choices':[{'delta':{},'finish_reason':'stop'}]},
                {'choices':[],'usage':{'completion_tokens':3,'prompt_tokens':20}}]
        text=''.join('data: '+json.dumps(e)+'\n\n' for e in events)
        if not truncated:text+='data: [DONE]\n\n'
        return httpx.Response(200,text=text)
    return httpx.MockTransport(handler)

def test_sse_usage_not_chunk_count():
    with VLMClient(transport=transport()) as c:r=c.complete([user_message('描述',[])])
    assert r.text=='这是图片' and r.completion_tokens==3 and r.content_chunks==1
    assert r.tpot_s is None and r.prompt_tokens==20

def test_truncated_stream_fails():
    with VLMClient(transport=transport(True)) as c:
        with pytest.raises(RuntimeError):c.complete([user_message('描述',[])])

def test_empty_question():
    with pytest.raises(ValueError):user_message('  ',[])
