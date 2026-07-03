from typing import Any

from fastapi import Body, FastAPI, Query
from fastapi.responses import HTMLResponse

from app.schemas import ChatRequest, ChatResponse, Message
from app.services.chat_controller import ChatController


app = FastAPI(title="SHL Assessment Recommender")
controller = ChatController()


CHAT_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SHL Assessment Recommender</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }
    textarea { width: 100%; min-height: 120px; font-size: 16px; padding: 12px; box-sizing: border-box; }
    button { margin-top: 12px; margin-right: 8px; padding: 10px 16px; font-size: 16px; cursor: pointer; }
    pre { background: #f5f5f5; padding: 16px; overflow-x: auto; white-space: pre-wrap; }
    .turn { border-bottom: 1px solid #ddd; padding: 12px 0; }
    .role { font-weight: 700; }
    .reply { margin-top: 24px; }
  </style>
</head>
<body>
  <h1>SHL Assessment Recommender</h1>
  <textarea id="message">Hiring a Java developer who works with stakeholders</textarea>
  <br>
  <button onclick="sendMessage()">Ask</button>
  <button onclick="resetChat()">Reset</button>
  <div class="reply">
    <h2>Conversation</h2>
    <div id="conversation">No messages yet.</div>
  </div>
  <div class="reply">
    <h2>Latest API Response</h2>
    <pre id="response">Type a hiring need and click Ask.</pre>
  </div>
  <script>
    const messages = [];

    function renderConversation() {
      const conversation = document.getElementById("conversation");
      if (messages.length === 0) {
        conversation.textContent = "No messages yet.";
        return;
      }
      conversation.innerHTML = messages.map(message => (
        `<div class="turn"><span class="role">${message.role}:</span> ${escapeHtml(message.content)}</div>`
      )).join("");
    }

    function escapeHtml(value) {
      return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    async function sendMessage() {
      const content = document.getElementById("message").value;
      const responseBox = document.getElementById("response");
      if (!content.trim()) {
        return;
      }
      messages.push({role: "user", content});
      renderConversation();
      responseBox.textContent = "Loading...";
      const response = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({messages})
      });
      const data = await response.json();
      if (data.reply) {
        messages.push({role: "assistant", content: data.reply});
        renderConversation();
      }
      responseBox.textContent = JSON.stringify(data, null, 2);
      document.getElementById("message").value = "";
    }

    function resetChat() {
      messages.length = 0;
      renderConversation();
      document.getElementById("response").textContent = "Type a hiring need and click Ask.";
      document.getElementById("message").value = "";
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return CHAT_PAGE


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return controller.respond(request.messages)


@app.post("/ask", response_model=ChatResponse)
def ask(content: str = Body(..., media_type="text/plain")) -> ChatResponse:
    return controller.respond([Message(role="user", content=content)])


@app.get("/debug/retrieval", include_in_schema=False)
def retrieval_debug(query: str = Query(..., min_length=3)) -> dict[str, Any]:
    return controller.recommender.retrieval_debug(query)
