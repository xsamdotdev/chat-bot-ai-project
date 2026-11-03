import { useCallback, useEffect, useState } from "react";
import { ChatView } from "./components/ChatView";
import { Composer } from "./components/Composer";
import { Sidebar } from "./components/Sidebar";
import { MenuIcon } from "./components/icons";
import { getHealth, listDocuments, sendChat } from "./lib/api";
import { initialOf } from "./lib/format";
import type { ChatMessage, DocumentInfo, HealthResponse } from "./lib/types";

const HISTORY_LIMIT = 12;
let idCounter = 0;

function nextId(): string {
  idCounter += 1;
  return `m-${Date.now()}-${idCounter}`;
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const refresh = useCallback(async () => {
    const [healthResult, documentsResult] = await Promise.allSettled([
      getHealth(),
      listDocuments(),
    ]);

    if (healthResult.status === "fulfilled") {
      setHealth(healthResult.value);
      setConnectionError(null);
    } else {
      const reason: unknown = healthResult.reason;
      setConnectionError(
        reason instanceof Error ? reason.message : "The backend is unreachable.",
      );
    }

    if (documentsResult.status === "fulfilled") {
      setDocuments(documentsResult.value.documents);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const botName = health?.chatbot_name ?? "Lisa";

  async function handleSend(text: string) {
    const query = text.trim();
    if (!query || sending) return;

    setDraft("");
    setSending(true);
    setSidebarOpen(false);

    const history = messages
      .filter((message) => !message.pending && !message.error && message.content)
      .slice(-HISTORY_LIMIT)
      .map((message) => ({
        role: message.role,
        content: message.content.slice(0, 8000),
      }));

    const userMessage: ChatMessage = { id: nextId(), role: "user", content: query };
    const replyId = nextId();

    setMessages((previous) => [
      ...previous,
      userMessage,
      { id: replyId, role: "assistant", content: "", pending: true },
    ]);

    try {
      const reply = await sendChat(query, history);
      setMessages((previous) =>
        previous.map((message) =>
          message.id === replyId
            ? { ...message, pending: false, content: reply.answer }
            : message,
        ),
      );
    } catch (cause) {
      const detail = cause instanceof Error ? cause.message : "Something went wrong.";
      setMessages((previous) =>
        previous.map((message) =>
          message.id === replyId
            ? { id: replyId, role: "assistant", content: detail, error: true }
            : message,
        ),
      );
    } finally {
      setSending(false);
    }
  }

  function startNewChat() {
    setMessages([]);
    setDraft("");
    setSidebarOpen(false);
  }

  return (
    <div className="app">
      <Sidebar
        documents={documents}
        health={health}
        botName={botName}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onDocumentsChanged={() => {
          void refresh();
        }}
      />

      {sidebarOpen && (
        <button
          className="scrim"
          type="button"
          aria-label="Close sidebar"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <main className="main">
        <header className="topbar">
          <button
            className="icon-button menu-button"
            type="button"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open documents sidebar"
          >
            <MenuIcon />
          </button>

          <div className="identity">
            <span className="avatar avatar-sm" aria-hidden="true">
              {initialOf(botName)}
            </span>
            <div>
              <p className="identity-name">{botName}</p>
              <p className="identity-status">
                <span className={`status-dot ${health ? "online" : "offline"}`} />
                {health ? "Online" : "Offline"}
              </p>
            </div>
          </div>

          <button
            className="ghost-button"
            type="button"
            onClick={startNewChat}
            disabled={messages.length === 0}
          >
            New chat
          </button>
        </header>

        {connectionError && (
          <p className="connection-banner" role="alert">
            {connectionError}
          </p>
        )}

        <ChatView
          messages={messages}
          botName={botName}
          onSuggestion={(suggestion) => {
            void handleSend(suggestion);
          }}
        />

        <Composer
          value={draft}
          botName={botName}
          disabled={sending}
          onChange={setDraft}
          onSend={(text) => {
            void handleSend(text);
          }}
        />
      </main>
    </div>
  );
}
