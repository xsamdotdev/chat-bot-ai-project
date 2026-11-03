import { useMemo, useRef, useState } from "react";
import { deleteDocument, uploadDocument } from "../lib/api";
import { formatBytes, formatDate, initialOf } from "../lib/format";
import type { DocumentInfo, HealthResponse } from "../lib/types";
import { CloseIcon, TrashIcon, UploadIcon } from "./icons";

const ACCEPTED = [".pdf", ".docx", ".txt", ".md"];
const MAX_UPLOAD_MB = 50;

interface SidebarProps {
  documents: DocumentInfo[];
  health: HealthResponse | null;
  botName: string;
  open: boolean;
  onClose: () => void;
  onDocumentsChanged: () => void;
}

export function Sidebar({
  documents,
  health,
  botName,
  open,
  onClose,
  onDocumentsChanged,
}: SidebarProps) {
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const totalChunks = useMemo(
    () => documents.reduce((sum, document) => sum + document.chunks, 0),
    [documents],
  );

  async function handleUpload(file: File) {
    setError(null);
    setNotice(null);

    const extension = `.${file.name.split(".").pop()?.toLowerCase() ?? ""}`;
    if (!ACCEPTED.includes(extension)) {
      setError(`Unsupported file type. Use ${ACCEPTED.join(", ")}.`);
      return;
    }
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setError(`"${file.name}" is larger than ${MAX_UPLOAD_MB} MB.`);
      return;
    }

    setUploading(true);
    try {
      await uploadDocument(file);
      setNotice(`"${file.name}" is ready — ask me about it!`);
      onDocumentsChanged();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Upload failed.");
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  async function handleDelete(document: DocumentInfo) {
    if (!window.confirm(`Remove "${document.filename}"?`)) return;

    setDeletingId(document.doc_id);
    setError(null);
    setNotice(null);
    try {
      await deleteDocument(document.doc_id);
      setNotice(`"${document.filename}" was removed.`);
      onDocumentsChanged();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Delete failed.");
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <aside className={`sidebar${open ? " open" : ""}`} aria-label="Documents">
      <div className="sidebar-header">
        <div className="brand">
          <span className="avatar avatar-md" aria-hidden="true">
            {initialOf(botName)}
          </span>
          <div>
            <p className="brand-name">{botName}</p>
            <p className="brand-sub">AI document assistant</p>
          </div>
        </div>
        <button
          className="icon-button sidebar-close"
          type="button"
          onClick={onClose}
          aria-label="Close sidebar"
        >
          <CloseIcon />
        </button>
      </div>

      <label
        className={`upload-zone${dragging ? " dragging" : ""}${uploading ? " busy" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const file = event.dataTransfer.files?.[0];
          if (file && !uploading) void handleUpload(file);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void handleUpload(file);
          }}
        />
        <UploadIcon className="upload-icon" />
        <span className="upload-title">
          {uploading ? "Uploading…" : dragging ? "Drop it here" : "Add a document"}
        </span>
        <span className="upload-sub">
          PDF, DOCX, TXT, MD · up to {MAX_UPLOAD_MB} MB
        </span>
      </label>

      {(error ?? notice) && (
        <p
          className={`sidebar-message${error ? " error" : ""}`}
          role={error ? "alert" : "status"}
        >
          {error ?? notice}
        </p>
      )}

      <div className="documents-header">
        <h2>Documents</h2>
        <span className="badge">{documents.length}</span>
      </div>

      <div className="document-list">
        {documents.length === 0 ? (
          <p className="empty-documents">
            Nothing here yet. Add a document and I'll be ready to help.
          </p>
        ) : (
          documents.map((document) => (
            <article className="document-card" key={document.doc_id}>
              <div className="document-main">
                <p className="document-name" title={document.filename}>
                  {document.filename}
                </p>
                <p className="document-meta">
                  {document.chunks} chunk{document.chunks === 1 ? "" : "s"} ·{" "}
                  {formatBytes(document.size_bytes)} · {formatDate(document.uploaded_at)}
                </p>
              </div>
              <button
                className="icon-button danger"
                type="button"
                onClick={() => void handleDelete(document)}
                disabled={deletingId === document.doc_id}
                aria-label={`Remove ${document.filename}`}
              >
                <TrashIcon />
              </button>
            </article>
          ))
        )}
      </div>

      <footer className="sidebar-footer">
        <div className="health">
          <span className={`status-dot ${health ? "online" : "offline"}`} />
          <span>{health ? "Connected" : "Offline"}</span>
        </div>
        {health && (
          <dl className="health-details">
            <div>
              <dt>Chat</dt>
              <dd>{health.chat_model}</dd>
            </div>
            <div>
              <dt>Embeddings</dt>
              <dd>{health.embedding_model}</dd>
            </div>
            <div>
              <dt>Indexed chunks</dt>
              <dd>{totalChunks}</dd>
            </div>
          </dl>
        )}
      </footer>
    </aside>
  );
}
