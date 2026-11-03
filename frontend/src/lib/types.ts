export interface HealthResponse {
  status: string;
  provider: string;
  chatbot_name: string;
  chat_model: string;
  embedding_model: string;
  embedding_provider: string;
  vector_store: string;
  documents_indexed: number;
}

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  chunks: number;
  size_bytes: number;
  uploaded_at: string;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
  total: number;
  embedding_model: string;
}

export interface UploadResponse {
  message: string;
  document: DocumentInfo;
}

export interface DeleteResponse {
  message: string;
  doc_id: string;
  chunks_deleted: number;
}

export interface Source {
  doc_id?: string | null;
  filename: string;
  chunk_index?: number | null;
  page?: number | null;
  snippet: string;
  score?: number | null;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  provider: string;
  model: string;
  chatbot_name: string;
}

export type Role = "user" | "assistant";

export interface HistoryMessage {
  role: Role;
  content: string;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  pending?: boolean;
  error?: boolean;
}
