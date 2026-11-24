"""Start with python -m frontend.app from 02-Multimodal/phase2-sglang."""
import json
import time
import gradio as gr
import httpx
from api.client import VLMClient, user_message
from preprocessing.image_budget import prepare_image

def answer(files, question, strategy, history, display):
    history=list(history or []); display=list(display or [])
    try:
        if not history and not files: raise ValueError("首轮请上传至少一张图片")
        if len(history)>=20: raise ValueError("已达 10 轮上限，请清空会话")
        start=time.perf_counter()
        images=[prepare_image(f,strategy,question) for f in (files or [])]
        message=user_message(question,images)
        prep=time.perf_counter()-start
        with VLMClient() as client:
            result=client.complete(history+[message])
        history += [message,{"role":"assistant","content":result.text}]
        display += [{"role":"user","content":question+ (f"（新增 {len(images)} 张图）" if images else "")},
                    {"role":"assistant","content":result.text}]
        stats=result.to_dict(); stats.pop("text")
        stats.update(preprocessing_s=prep,images=[i.metadata for i in images])
        return display,history,json.dumps(stats,ensure_ascii=False,indent=2),None,""
    except (ValueError,OSError,RuntimeError,httpx.HTTPError) as exc:
        # Preserve successful history on failure; do not persist failed user turns.
        return display,history,"请求失败："+str(exc),files,question

def build_app():
    with gr.Blocks(title="SGLang 视觉语言助手") as app:
        gr.Markdown("# 多模态视觉问答")
        state=gr.State([])
        files=gr.File(file_count="multiple",file_types=["image"],type="filepath",label="图片（最多 8 张）")
        strategy=gr.Dropdown(
            ["baseline", "low", "medium", "high", ("自适应（改进规则）", "adaptive_v2"),
             ("自适应（旧规则）", "adaptive")],
            value="baseline", label="本轮新增图片预算",
        )
        chat=gr.Chatbot(label="多轮会话")
        question=gr.Textbox(label="问题",placeholder="请描述图片，或读取图中的文字")
        stats=gr.Textbox(label="性能信息",lines=8)
        send=gr.Button("提问",variant="primary"); clear=gr.Button("清空会话")
        send.click(answer,[files,question,strategy,state,chat],[chat,state,stats,files,question])
        clear.click(lambda:([],[],"",None,""),outputs=[chat,state,stats,files,question])
    return app

if __name__=="__main__":
    build_app().queue().launch(server_name="127.0.0.1",share=False)
