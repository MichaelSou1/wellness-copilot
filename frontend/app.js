const $ = (id) => document.getElementById(id);

const els = {
  apiKey: $("apiKey"),
  bindBtn: $("bindBtn"),
  bindingForm: $("bindingForm"),
  bindingState: $("bindingState"),
  bindingsList: $("bindingsList"),
  chatForm: $("chatForm"),
  chatState: $("chatState"),
  clearChatBtn: $("clearChatBtn"),
  displayName: $("displayName"),
  healthStatus: $("healthStatus"),
  imageInput: $("imageInput"),
  imagePreview: $("imagePreview"),
  latencyMetric: $("latencyMetric"),
  loginState: $("loginState"),
  messageInput: $("messageInput"),
  messages: $("messages"),
  newThreadBtn: $("newThreadBtn"),
  previewImg: $("previewImg"),
  qrHint: $("qrHint"),
  qrImage: $("qrImage"),
  qrLoginBtn: $("qrLoginBtn"),
  qrPanel: $("qrPanel"),
  projectUserId: $("projectUserId"),
  refreshBindingsBtn: $("refreshBindingsBtn"),
  removeImageBtn: $("removeImageBtn"),
  retryJobsBtn: $("retryJobsBtn"),
  routeMetric: $("routeMetric"),
  sendBtn: $("sendBtn"),
  threadId: $("threadId"),
  userId: $("userId"),
  visionMetric: $("visionMetric"),
  wechatWxid: $("wechatWxid"),
};

const storageKeys = {
  apiKey: "wellness.web.apiKey",
  userId: "wellness.web.userId",
  threadId: "wellness.web.threadId",
};

const state = {
  busy: false,
  config: {
    api_key_required: false,
    max_image_bytes: 8 * 1024 * 1024,
  },
  image: null,
  lastJobId: "",
  qrPolling: false,
};

function readStorage(key, fallback = "") {
  try {
    return localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
}

function writeStorage(key, value) {
  try {
    localStorage.setItem(key, value || "");
  } catch {
    return;
  }
}

function setStatus(el, text, kind = "") {
  el.textContent = text;
  el.classList.remove("status-ok", "status-warn", "status-error");
  if (kind) el.classList.add(`status-${kind}`);
}

function authHeaders() {
  const headers = { "Content-Type": "application/json" };
  const key = els.apiKey.value.trim();
  if (key) headers["X-API-Key"] = key;
  return headers;
}

function missingApiKey() {
  return state.config.api_key_required && !els.apiKey.value.trim();
}

function errorMessage(error) {
  if (!error) return "请求失败";
  if (typeof error === "string") return error;
  if (error.message) return error.message;
  return "请求失败";
}

async function parseApiError(response) {
  let detail = "";
  try {
    const payload = await response.json();
    const raw = payload.detail || payload.error || payload.message || "";
    if (typeof raw === "string") detail = raw;
    else if (raw.message) detail = raw.message;
    else if (raw.error) detail = raw.error;
    else detail = JSON.stringify(raw);
  } catch {
    detail = await response.text();
  }
  return new Error(detail || `${response.status} ${response.statusText}`);
}

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...authHeaders(),
      ...(options.headers || {}),
    },
  });
  if (!response.ok) throw await parseApiError(response);
  return response.json();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setBusy(value) {
  state.busy = value;
  els.sendBtn.disabled = value;
  els.bindBtn.disabled = value;
  els.qrLoginBtn.disabled = value || state.qrPolling;
}

function clearEmptyState() {
  const empty = els.messages.querySelector(".empty-state");
  if (empty) empty.remove();
}

function ensureEmptyState() {
  if (els.messages.children.length > 0) return;
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.textContent = "今天想处理什么？";
  els.messages.appendChild(empty);
}

function appendMessage(role, text, options = {}) {
  clearEmptyState();
  const node = document.createElement("article");
  node.className = `message ${role}`;

  if (options.imageUrl) {
    const img = document.createElement("img");
    img.src = options.imageUrl;
    img.alt = "聊天图片";
    node.appendChild(img);
  }

  const body = document.createElement("div");
  body.className = "message-text";
  body.textContent = text || "";
  node.appendChild(body);

  if (options.meta) {
    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = options.meta;
    node.appendChild(meta);
  }

  els.messages.appendChild(node);
  els.messages.scrollTop = els.messages.scrollHeight;
  return node;
}

function appendProgressMessage() {
  clearEmptyState();
  const node = document.createElement("article");
  node.className = "message system progress-message";

  const body = document.createElement("div");
  body.className = "message-text";
  body.textContent = "正在启动处理流程";

  const steps = document.createElement("div");
  steps.className = "progress-steps";

  const detail = document.createElement("div");
  detail.className = "message-meta";
  detail.textContent = "等待首个节点事件";

  node.append(body, steps, detail);
  els.messages.appendChild(node);
  els.messages.scrollTop = els.messages.scrollHeight;
  return { node, body, steps, detail, seen: new Set(), current: "" };
}

function nodeLabel(raw) {
  const map = {
    InputAccumulator: "输入聚合",
    TurnStart: "画像/记忆",
    QueryRewriter: "问题改写",
    MultiModalPreprocessor: "图片预处理",
    Orchestrator: "专家调度",
    Aggregator: "汇总回答",
    Critic: "安全审核",
    ReplanJudge: "复核",
    FakeAgent: "测试代理",
  };
  return map[raw] || raw || "处理中";
}

function updateProgress(progress, eventName, data = {}) {
  const nodeName = data.node || "";
  if (nodeName) {
    progress.current = nodeName;
    const key = `${nodeName}`;
    let chip = progress.steps.querySelector(`[data-node-key="${key}"]`);
    if (!chip) {
      chip = document.createElement("span");
      chip.className = "step-chip";
      chip.dataset.nodeKey = key;
      chip.textContent = nodeLabel(nodeName);
      progress.steps.appendChild(chip);
    }
    for (const item of progress.steps.querySelectorAll(".step-chip")) {
      item.classList.toggle("active", item.dataset.nodeKey === key);
    }
    if (eventName === "node_output") {
      chip.classList.remove("active");
      chip.classList.add("done");
    }
    progress.body.textContent = eventName === "node_output"
      ? `${nodeLabel(nodeName)} 已完成`
      : `正在处理：${nodeLabel(nodeName)}`;
  }

  const bits = [];
  if (data.orchestrator_decision) bits.push(`调度：${data.orchestrator_decision}`);
  if (data.critic_verdict) bits.push(`审核：${data.critic_verdict}`);
  if (data.tools && data.tools.length) bits.push(`工具：${data.tools.slice(0, 3).join(", ")}`);
  if (Number(data.retrieval_hits || 0) > 0) bits.push(`检索 ${data.retrieval_hits} 条`);
  if (Number(data.vision_calls || 0) > 0) bits.push("已看图");
  if (data.answer_preview && !bits.length) bits.push(data.answer_preview);
  progress.detail.textContent = bits.join(" · ") || "正在等待下一步";
  els.messages.scrollTop = els.messages.scrollHeight;
}

function finishProgress(progress, result) {
  updateThread(result.thread_id);
  updateMetrics(result);
  progress.node.className = "message assistant";
  progress.body.textContent = result.answer || "没有返回内容";
  progress.steps.remove();
  progress.detail.textContent = result.trace_id ? `trace ${result.trace_id}` : "";
  setStatus(els.chatState, "就绪", "ok");
  els.messages.scrollTop = els.messages.scrollHeight;
}

function updateThread(threadId) {
  if (!threadId) return;
  els.threadId.value = threadId;
  writeStorage(storageKeys.threadId, threadId);
}

function updateMetrics(result) {
  els.routeMetric.textContent = result.route || "-";
  els.latencyMetric.textContent = result.latency_ms ? `${(result.latency_ms / 1000).toFixed(1)}s` : "-";
  els.visionMetric.textContent = Number(result.vision_calls || 0).toString();
}

function handleFinalResult(result) {
  updateThread(result.thread_id);
  updateMetrics(result);
  appendMessage("assistant", result.answer || "没有返回内容", {
    meta: result.trace_id ? `trace ${result.trace_id}` : "",
  });
  setStatus(els.chatState, "就绪", "ok");
}

async function pollJob(jobId) {
  state.lastJobId = jobId;
  setStatus(els.chatState, "处理中", "warn");
  for (let i = 0; i < 180; i += 1) {
    const job = await apiFetch(`/v1/jobs/${encodeURIComponent(jobId)}`, { method: "GET" });
    if (job.thread_id) updateThread(job.thread_id);
    if (job.status === "succeeded") {
      handleFinalResult(job.result || {});
      return;
    }
    if (job.status === "dead") {
      throw new Error(job.error || "任务失败");
    }
    await sleep(2000);
  }
  throw new Error("任务仍在处理中");
}

async function handleChatResponse(payload) {
  if (payload.answer) {
    handleFinalResult(payload);
    return;
  }
  if (payload.job_id) {
    appendMessage("system", "任务处理中", { meta: payload.job_id });
    await pollJob(payload.job_id);
    return;
  }
  appendMessage("assistant", JSON.stringify(payload, null, 2));
}

function parseSseBlock(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      event = line.slice(6).trim() || event;
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  const rawData = dataLines.join("\n");
  let data = {};
  if (rawData) {
    try {
      data = JSON.parse(rawData);
    } catch {
      data = { raw: rawData };
    }
  }
  return { event, data };
}

async function streamChat(payload, progress) {
  const response = await fetch("/v1/chat/stream", {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw await parseApiError(response);
  if (!response.body) throw new Error("浏览器不支持流式读取");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult = null;

  while (true) {
    const { value, done } = await reader.read();
    if (value) {
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split(/\n\n/);
      buffer = blocks.pop() || "";
      for (const block of blocks) {
        const item = parseSseBlock(block);
        if (item.event === "final") {
          finalResult = item.data;
          finishProgress(progress, item.data);
        } else if (item.event === "error") {
          throw new Error(item.data.detail || item.data.error || "流式处理失败");
        } else {
          updateProgress(progress, item.event, item.data);
        }
      }
    }
    if (done) break;
  }

  if (buffer.trim()) {
    const item = parseSseBlock(buffer);
    if (item.event === "final") {
      finalResult = item.data;
      finishProgress(progress, item.data);
    } else if (item.event === "error") {
      throw new Error(item.data.detail || item.data.error || "流式处理失败");
    }
  }
  if (!finalResult) throw new Error("流式处理未返回最终结果");
  return finalResult;
}

function renderLoginStatus(login) {
  if (login && login.configured) {
    const source = login.source === "runtime" ? "扫码保存" : "环境变量";
    setStatus(els.loginState, `已登录 · ${source}`, "ok");
    els.qrPanel.classList.add("hidden");
    return;
  }
  setStatus(els.loginState, "未登录", "warn");
}

async function loadWechatLoginStatus() {
  if (missingApiKey()) {
    setStatus(els.loginState, "需要 API Key", "warn");
    return;
  }
  try {
    const login = await apiFetch("/v1/wechat/login/status", { method: "GET" });
    renderLoginStatus(login);
  } catch (error) {
    setStatus(els.loginState, errorMessage(error), "error");
  }
}

async function pollWechatQr(qrcode, pollBaseUrl = "") {
  state.qrPolling = true;
  els.qrLoginBtn.disabled = true;
  let nextPollBaseUrl = pollBaseUrl;
  try {
    for (let i = 0; i < 90; i += 1) {
      const result = await apiFetch("/v1/wechat/login/poll", {
        method: "POST",
        body: JSON.stringify({
          qrcode,
          poll_base_url: nextPollBaseUrl,
        }),
      });
      if (result.next_poll_base_url) {
        nextPollBaseUrl = result.next_poll_base_url;
      }
      if (result.authorized) {
        renderLoginStatus(result.login || { configured: true, source: "runtime" });
        setStatus(els.bindingState, "微信已登录", "ok");
        return;
      }
      const statusText = result.state && result.state !== "pending" ? result.state : "等待扫码";
      setStatus(els.loginState, statusText, "warn");
      await sleep(2000);
    }
    setStatus(els.loginState, "二维码已超时", "error");
  } catch (error) {
    setStatus(els.loginState, errorMessage(error), "error");
  } finally {
    state.qrPolling = false;
    els.qrLoginBtn.disabled = state.busy;
  }
}

async function startWechatLogin() {
  if (missingApiKey()) {
    setStatus(els.loginState, "需要 API Key", "warn");
    els.apiKey.focus();
    return;
  }
  state.qrPolling = true;
  els.qrLoginBtn.disabled = true;
  setStatus(els.loginState, "获取二维码", "warn");
  try {
    writeStorage(storageKeys.apiKey, els.apiKey.value.trim());
    const qr = await apiFetch("/v1/wechat/login/qrcode", { method: "POST", body: "{}" });
    if (qr.image_data_url) {
      els.qrImage.src = qr.image_data_url;
      els.qrPanel.classList.remove("hidden");
      els.qrHint.textContent = "请用微信扫码，并在手机上确认授权。";
    } else if (qr.qrcode_url || qr.payload) {
      els.qrPanel.classList.remove("hidden");
      els.qrImage.removeAttribute("src");
      els.qrHint.textContent = qr.qrcode_url || qr.payload;
    }
    setStatus(els.loginState, "等待扫码", "warn");
    await pollWechatQr(qr.qrcode);
  } catch (error) {
    state.qrPolling = false;
    els.qrLoginBtn.disabled = state.busy;
    setStatus(els.loginState, errorMessage(error), "error");
  }
}

function resetImage() {
  state.image = null;
  els.imageInput.value = "";
  els.previewImg.removeAttribute("src");
  els.imagePreview.classList.add("hidden");
}

function setImage(file, dataUrl) {
  state.image = {
    data: dataUrl,
    mime_type: file.type || "image/jpeg",
    filename: file.name || "",
    previewUrl: dataUrl,
  };
  els.previewImg.src = dataUrl;
  els.imagePreview.classList.remove("hidden");
}

function autosizeTextarea() {
  els.messageInput.style.height = "auto";
  els.messageInput.style.height = `${Math.min(160, els.messageInput.scrollHeight)}px`;
}

async function sendChat(event) {
  event.preventDefault();
  if (state.busy) return;
  const message = els.messageInput.value.trim();
  if (!message) {
    setStatus(els.chatState, "请输入文本", "warn");
    els.messageInput.focus();
    return;
  }

  const image = state.image;
  appendMessage("user", message, { imageUrl: image ? image.previewUrl : "" });
  els.messageInput.value = "";
  autosizeTextarea();
  resetImage();
  setBusy(true);
  setStatus(els.chatState, "发送中", "warn");
  let progress = null;

  try {
    writeStorage(storageKeys.apiKey, els.apiKey.value.trim());
    writeStorage(storageKeys.userId, els.userId.value.trim());
    writeStorage(storageKeys.threadId, els.threadId.value.trim());
    const payload = {
      user_id: els.userId.value.trim() || "default_user",
      thread_id: els.threadId.value.trim(),
      message,
      source: "web",
    };
    if (image) {
      payload.image = {
        data: image.data,
        mime_type: image.mime_type,
        filename: image.filename,
      };
    }
    progress = appendProgressMessage();
    setStatus(els.chatState, "流式处理中", "warn");
    try {
      await streamChat(payload, progress);
    } catch (streamError) {
      updateProgress(progress, "fallback", { answer_preview: "流式连接中断，切换到普通请求" });
      const result = await apiFetch("/v1/chat", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      progress.node.remove();
      await handleChatResponse(result);
    }
  } catch (error) {
    if (progress && progress.node.isConnected) {
      progress.node.remove();
    }
    appendMessage("error", errorMessage(error));
    setStatus(els.chatState, "失败", "error");
  } finally {
    setBusy(false);
  }
}

async function loadBindings() {
  if (missingApiKey()) {
    els.bindingsList.innerHTML = "";
    setStatus(els.bindingState, "需要 API Key", "warn");
    return;
  }
  setStatus(els.bindingState, "加载中", "warn");
  try {
    const data = await apiFetch("/v1/wechat/bindings", { method: "GET" });
    renderBindings(data.bindings || []);
    setStatus(els.bindingState, `${(data.bindings || []).length} 条`, "ok");
  } catch (error) {
    els.bindingsList.innerHTML = "";
    setStatus(els.bindingState, errorMessage(error), "error");
  }
}

function renderBindings(bindings) {
  els.bindingsList.innerHTML = "";
  if (!bindings.length) {
    const empty = document.createElement("div");
    empty.className = "binding-item";
    empty.textContent = "暂无绑定";
    els.bindingsList.appendChild(empty);
    return;
  }
  const formatter = new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
  for (const row of bindings) {
    const item = document.createElement("div");
    item.className = "binding-item";

    const title = document.createElement("div");
    title.className = "binding-title";
    const wxid = document.createElement("span");
    wxid.textContent = row.wechat_wxid || "";
    const user = document.createElement("span");
    user.textContent = row.project_user_id || "";
    title.append(wxid, user);

    const meta = document.createElement("div");
    meta.className = "binding-meta";
    const updated = row.updated_at ? formatter.format(new Date(Number(row.updated_at) * 1000)) : "";
    meta.textContent = [row.display_name || "", updated].filter(Boolean).join(" · ");

    item.append(title, meta);
    els.bindingsList.appendChild(item);
  }
}

async function submitBinding(event) {
  event.preventDefault();
  const wxid = els.wechatWxid.value.trim();
  if (!wxid) {
    setStatus(els.bindingState, "wxid 为空", "warn");
    els.wechatWxid.focus();
    return;
  }
  setBusy(true);
  setStatus(els.bindingState, "保存中", "warn");
  try {
    writeStorage(storageKeys.apiKey, els.apiKey.value.trim());
    await apiFetch("/v1/wechat/bindings", {
      method: "POST",
      body: JSON.stringify({
        wechat_wxid: wxid,
        user_id: els.projectUserId.value.trim(),
        display_name: els.displayName.value.trim(),
      }),
    });
    els.bindingForm.reset();
    await loadBindings();
  } catch (error) {
    setStatus(els.bindingState, errorMessage(error), "error");
  } finally {
    setBusy(false);
  }
}

async function loadConfigAndHealth() {
  try {
    const config = await fetch("/v1/frontend/config").then((res) => res.json());
    state.config = { ...state.config, ...config };
    if (config.api_key_required && !els.apiKey.value.trim()) {
      setStatus(els.loginState, "需要 API Key", "warn");
      setStatus(els.bindingState, "需要 API Key", "warn");
    }
  } catch {
    setStatus(els.healthStatus, "配置不可用", "error");
  }

  try {
    const health = await fetch("/healthz").then((res) => res.json());
    setStatus(els.healthStatus, health.ok ? "在线" : "异常", health.ok ? "ok" : "error");
  } catch {
    setStatus(els.healthStatus, "离线", "error");
  }
}

function loadPreferences() {
  els.apiKey.value = readStorage(storageKeys.apiKey);
  els.userId.value = readStorage(storageKeys.userId, "default_user");
  els.threadId.value = readStorage(storageKeys.threadId);
}

function bindEvents() {
  els.chatForm.addEventListener("submit", sendChat);
  els.bindingForm.addEventListener("submit", submitBinding);
  els.qrLoginBtn.addEventListener("click", startWechatLogin);
  els.refreshBindingsBtn.addEventListener("click", loadBindings);
  els.clearChatBtn.addEventListener("click", () => {
    els.messages.innerHTML = "";
    ensureEmptyState();
  });
  els.newThreadBtn.addEventListener("click", () => {
    els.threadId.value = "";
    writeStorage(storageKeys.threadId, "");
    setStatus(els.chatState, "新会话", "ok");
  });
  els.retryJobsBtn.addEventListener("click", async () => {
    if (!state.lastJobId || state.busy) return;
    setBusy(true);
    try {
      await pollJob(state.lastJobId);
    } catch (error) {
      appendMessage("error", errorMessage(error));
    } finally {
      setBusy(false);
    }
  });
  els.removeImageBtn.addEventListener("click", resetImage);
  els.messageInput.addEventListener("input", autosizeTextarea);
  for (const input of [els.apiKey, els.userId, els.threadId]) {
    input.addEventListener("change", () => {
      writeStorage(storageKeys.apiKey, els.apiKey.value.trim());
      writeStorage(storageKeys.userId, els.userId.value.trim());
      writeStorage(storageKeys.threadId, els.threadId.value.trim());
      if (input === els.apiKey && els.apiKey.value.trim()) {
        loadWechatLoginStatus();
        loadBindings();
      }
    });
  }
  els.imageInput.addEventListener("change", () => {
    const file = els.imageInput.files && els.imageInput.files[0];
    if (!file) {
      resetImage();
      return;
    }
    if (file.size > state.config.max_image_bytes) {
      resetImage();
      setStatus(els.chatState, "图片过大", "error");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setImage(file, String(reader.result || ""));
    reader.onerror = () => setStatus(els.chatState, "图片读取失败", "error");
    reader.readAsDataURL(file);
  });
}

async function boot() {
  loadPreferences();
  bindEvents();
  ensureEmptyState();
  await loadConfigAndHealth();
  await loadWechatLoginStatus();
  await loadBindings();
}

boot();
