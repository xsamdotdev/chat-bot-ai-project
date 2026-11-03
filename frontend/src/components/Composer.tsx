import { useEffect, useRef } from "react";
import { SendIcon } from "./icons";

interface ComposerProps {
  value: string;
  botName: string;
  disabled: boolean;
  onChange: (value: string) => void;
  onSend: (value: string) => void;
}

export function Composer({ value, botName, disabled, onChange, onSend }: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 160)}px`;
  }, [value]);

  function submit() {
    if (disabled || !value.trim()) return;
    onSend(value);
  }

  return (
    <div className="composer">
      <form
        className="composer-box"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          maxLength={4000}
          placeholder={`Ask ${botName} anything…`}
          aria-label="Your message"
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
        />
        <button
          type="submit"
          className="send-button"
          disabled={disabled || !value.trim()}
          aria-label="Send message"
        >
          <SendIcon />
        </button>
      </form>
      <p className="composer-hint">Enter to send · Shift + Enter for a new line</p>
    </div>
  );
}
