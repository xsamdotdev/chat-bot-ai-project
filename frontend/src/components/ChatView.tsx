import { useEffect, useRef } from "react";
import { initialOf } from "../lib/format";
import type { ChatMessage } from "../lib/types";
import { MessageBubble } from "./MessageBubble";

const SUGGESTIONS = [
  "What can you help me with?",
  "Summarize the key points",
  "What are the most important details?",
  "Explain that in simple terms",
];

interface ChatViewProps {
  messages: ChatMessage[];
  botName: string;
  onSuggestion: (suggestion: string) => void;
}

export function ChatView({ messages, botName, onSuggestion }: ChatViewProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <section className="chat-scroll">
        <div className="welcome">
          <span className="avatar avatar-xl" aria-hidden="true">
            {initialOf(botName)}
          </span>
          <h1>Hi, I'm {botName}</h1>
          <p className="welcome-subtitle">
            Your friendly AI assistant. Ask me anything and I'll help you find the
            answer.
          </p>
          <div className="suggestions">
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                className="suggestion"
                onClick={() => onSuggestion(suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </div>
          <p className="welcome-hint">
            Tip: add a document from the sidebar to ask about it.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="chat-scroll" aria-live="polite">
      <div className="messages">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} botName={botName} />
        ))}
        <div ref={endRef} />
      </div>
    </section>
  );
}
