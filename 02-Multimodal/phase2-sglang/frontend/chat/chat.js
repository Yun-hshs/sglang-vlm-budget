"use strict";

const $ = (id) => document.getElementById(id);
const state = { chat: null, controller: null, files: [], busy: false };
const compose = $("compose");
const question = $("question");

function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw Error(typeof payload.detail === "string" ? payload.detail : "处理请求失败，请重试");
  }
  return response;
}

function notice(message = "") {
  $("notice").textContent = message;
}

function sidebar(open) {
  document.querySelector(".shell").classList.toggle("show-sidebar", open);
  $("toggle-sidebar").setAttribute("aria-expanded", String(open));
}

function buttons() {
  $("send").disabled = state.busy || !question.value.trim();
  $("send").hidden = state.busy;
  $("stop").hidden = !state.busy;
  $("stop").disabled = !state.controller;
  $("new-chat").disabled = state.busy;
  $("delete-chat").disabled = state.busy;
  $("delete-chat").hidden = !state.chat;
  $("file").disabled = state.busy;
  for (const element of $("chat-list").querySelectorAll("button")) element.disabled = state.busy;
  for (const element of $("attachments").querySelectorAll("button")) element.disabled = state.busy;
}

function scroll() {
  $("viewport").scrollTop = $("viewport").scrollHeight;
}

function autoSize() {
  question.style.height = "auto";
  question.style.height = Math.min(120, question.scrollHeight) + "px";
  buttons();
}

function fileList() {
  $("attachments").replaceChildren();
  state.files.forEach(({file, url}, i) => {
    const item = make("div", "attachment");
    const img = make("img");
    img.src = url;
    img.alt = "";
    const remove = make("button", null, "×");
    remove.type = "button";
    remove.setAttribute("aria-label", "移除 " + file.name);
    remove.addEventListener("click", () => {
      URL.revokeObjectURL(state.files[i].url);
      state.files.splice(i, 1);
      fileList();
    });
    item.append(img, make("span", null, file.name), remove);
    $("attachments").append(item);
  });
  buttons();
}

function clearFiles() {
  for (const {url} of state.files) URL.revokeObjectURL(url);
  state.files = [];
  fileList();
}

function statsLabel(stats) {
  const base = ["首字 " + stats.ttft_ms + " ms", "估计未缓存 " + stats.uncached_tokens_est + " Token"];
  if (stats.cached_tokens != null) base.push("实际前缀命中 " + stats.cached_tokens + " Token");
  if (stats.finish_reason === "length") base.push("已达到输出长度上限");
  return base.join(" · ");
}

function message(turn) {
  const article = make("article", "message " + turn.role);
  const bubble = make("div", "bubble");
  if (turn.role === "assistant") {
    const mark = make("div", "assistant-mark", "✳");
    mark.setAttribute("aria-hidden", "true");
    article.append(mark);
  }
  const body = make("div", "body");
  if (Array.isArray(turn.content)) {
    const images = make("div", "message-images");
    for (const piece of turn.content) {
      if (piece.type === "image_url") {
        const img = make("img");
        img.src = piece.image_url.url;
        img.alt = "对话附件";
        img.loading = "lazy";
        images.append(img);
      } else if (piece.type === "text") body.append(document.createTextNode(piece.text));
    }
    if (images.childElementCount) bubble.append(images);
  } else {
    body.textContent = turn.content;
  }
  bubble.append(body);
  if (turn.stats) bubble.append(make("div", "stats", statsLabel(turn.stats)));
  article.append(bubble);
  $("messages").append(article);
  $("welcome").hidden = true;
  return { article, body, bubble };
}

async function refresh() {
  const chats = await (await api("/api/chats")).json();
  $("chat-list").replaceChildren();
  if (!chats.length) $("chat-list").append(make("span", "empty-list", "还没有对话记录"));
  for (const chat of chats) {
    const selected = chat.id === state.chat;
    const button = make("button", selected ? "chat-item selected" : "chat-item", chat.title);
    button.title = chat.title;
    button.setAttribute("aria-current", selected ? "true" : "false");
    button.addEventListener("click", () => view(chat.id).catch((error) => notice(error.message)));
    $("chat-list").append(button);
  }
  buttons();
}

async function view(id) {
  if (state.busy) return;
  const history = id ? (await (await api("/api/chats/" + id)).json()).history : [];
  state.chat = id;
  question.value = "";
  clearFiles();
  $("messages").replaceChildren();
  $("welcome").hidden = history.length > 0;
  for (const turn of history) message(turn);
  notice();
  autoSize();
  sidebar(false);
  await refresh();
  scroll();
  question.focus({preventScroll: true});
}

async function receive(response, handle) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const ndjson = response.headers.get("content-type")?.startsWith("application/x-ndjson");
  const separator = ndjson ? "\n" : "\n\n";
  let buffer = "";
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      let end;
      while ((end = buffer.indexOf(separator)) >= 0) {
        const line = buffer.slice(0, end);
        buffer = buffer.slice(end + separator.length);
        if (ndjson && line.trim()) handle(JSON.parse(line));
        else if (!ndjson && line.startsWith("data:")) handle(JSON.parse(line.slice(5)));
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
}

compose.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text || state.busy) return;
  state.busy = true;
  buttons();
  notice();
  let sent;
  let reply;
  let finished = false;
  try {
    if (!state.chat) state.chat = (await (await api("/api/chats", {method: "POST"})).json()).id;
    const data = new FormData();
    data.append("question", text);
    for (const {file} of state.files) data.append("images", file);
    const content = state.files.map(({url}) => ({type: "image_url", image_url: {url}}));
    sent = message({role: "user", content: [...content, {type: "text", text}]});
    reply = message({role: "assistant", content: "正在准备回答…"});
    reply.body.classList.add("pending");
    scroll();
    state.controller = new AbortController();
    buttons();
    const response = await api("/api/chats/" + state.chat + "/stream", {
      method: "POST", body: data, signal: state.controller.signal,
      headers: {Accept: "application/x-ndjson"},
    });
    await receive(response, (item) => {
      if (item.type === "queued") reply.body.textContent = "等待模型生成…";
      if (item.type === "delta") {
        if (reply.body.classList.contains("pending")) {
          reply.body.classList.remove("pending");
          reply.body.textContent = "";
        }
        reply.body.append(document.createTextNode(item.text));
        scroll();
      }
      if (item.type === "done") {
        finished = true;
        reply.bubble.append(make("div", "stats", statsLabel(item.stats)));
      }
      if (item.type === "error") throw Error(item.message);
    });
    if (!finished) throw Error("连接中断，请重试");
    question.value = "";
    // Current bubbles use object URLs. Use the persistent image URLs before revoking them.
    if (state.files.length) {
      const saved = await (await api("/api/chats/" + state.chat)).json();
      const parts = saved.history.at(-2).content.filter((piece) => piece.type === "image_url");
      sent.article.querySelectorAll("img").forEach((img, i) => { img.src = parts[i].image_url.url; });
    }
    clearFiles();
  } catch (error) {
    if (!finished) {
      sent?.article.remove();
      reply?.article.remove();
      $("welcome").hidden = $("messages").childElementCount > 0;
    }
    notice(error.name === "AbortError" ? "已停止生成，可编辑后重新发送" : error.message);
  } finally {
    state.busy = false;
    state.controller = null;
    autoSize();
    refresh().catch((error) => notice(error.message));
  }
});

$("stop").addEventListener("click", () => state.controller?.abort());
$("new-chat").addEventListener("click", () => view(null).catch((error) => notice(error.message)));
$("toggle-sidebar").addEventListener("click", () => sidebar($("toggle-sidebar").getAttribute("aria-expanded") !== "true"));
$("delete-chat").addEventListener("click", async () => {
  if (!state.chat || !window.confirm("删除这次对话及附件？")) return;
  try {
    await api("/api/chats/" + state.chat, {method: "DELETE"});
    await view(null);
  } catch (error) { notice(error.message); }
});
$("file").addEventListener("change", () => {
  notice();
  for (const file of $("file").files) {
    if (state.files.length >= 4) { notice("每轮最多 4 张图片"); break; }
    if (!file.type.startsWith("image/") || file.size > 8 * 1024 * 1024) {
      notice("请选择小于 8 MiB 的图片");
      continue;
    }
    state.files.push({file, url: URL.createObjectURL(file)});
  }
  $("file").value = "";
  fileList();
});
document.querySelector(".attach").addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    $("file").click();
  }
});
for (const button of document.querySelectorAll("[data-prompt]")) {
  button.addEventListener("click", () => {
    question.value = button.dataset.prompt;
    autoSize();
    question.focus();
  });
}
question.addEventListener("input", autoSize);
question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    compose.requestSubmit();
  }
});

async function health() {
  try {
    await api("/health");
    $("connection").textContent = "模型已连接";
    document.querySelector(".connection").classList.add("ready");
  } catch {
    $("connection").textContent = "模型暂未连接";
    document.querySelector(".connection").classList.remove("ready");
  }
}
health();
setInterval(health, 30000);
refresh().catch((error) => notice(error.message));
