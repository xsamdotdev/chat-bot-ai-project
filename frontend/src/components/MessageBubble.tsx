import { initialOf } from "../lib/format";
import type { ChatMessage } from "../lib/types";

interface MessageBubbleProps {
  message: ChatMessage;
  botName: string;
}

export function MessageBubble({ message, botName }: MessageBubbleProps) {
  if (message.role === "user") {
    return (
      <div className="message message-user">
        <div className="bubble bubble-user">{message.content}</div>
      </div>
    );
  }

  return (
    <div className="message message-bot">
      <span className="avatar avatar-sm" aria-hidden="true">
        {initialOf(botName)}
      </span>
      <div className={`bubble bubble-bot${message.error ? " bubble-error" : ""}`}>
        {message.pending ? (
          <span className="typing" role="status" aria-label={`${botName} is typing`}>
            <span />
            <span />
            <span />
          </span>
        ) : (
          <p className="bubble-text">{message.content}</p>
        )}
        {message.error && <span className="bubble-tag">Message not delivered</span>}
      </div>
    </div>
  );
}
