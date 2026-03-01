import { useState, useRef, useEffect } from 'react';
import { Bot, User, Send, Upload, FileText, CheckCircle2, AlertCircle, Loader2, Database, Sparkles, X } from 'lucide-react';
import './index.css';

function App() {
  const [messages, setMessages] = useState([
    { role: 'bot', content: 'Welcome! Upload your documents and I can answer questions about them using my RAG capabilities. Let me know what you want to learn.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadedDocs, setUploadedDocs] = useState([]);

  const chatAreaRef = useRef(null);

  const scrollToBottom = () => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTo({
        top: chatAreaRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch('http://localhost:8000/documents');
      if (res.ok) {
        const data = await res.json();
        setUploadedDocs(data.documents || []);
      }
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setUploadStatus(null);
    }
  };

  const removeFile = (e) => {
    e.stopPropagation();
    setFile(null);
    setUploadStatus(null);
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploadStatus({ type: 'loading', msg: 'Analyzing & indexing document...' });
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');

      setUploadStatus({ type: 'success', msg: 'Successfully added to knowledge base!' });
      setTimeout(() => {
        setFile(null);
        setUploadStatus(null);
      }, 3000);

      // Refresh document list
      fetchDocuments();

      // Auto-add a bot greeting indicating success
      setMessages(prev => [...prev, { role: 'bot', content: `I've successfully read "${file.name}". What would you like to know about it?` }]);
    } catch (err) {
      setUploadStatus({ type: 'error', msg: err.message });
    }
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMsg = input.trim();
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: userMsg })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to get answer');

      setMessages(prev => [...prev, { role: 'bot', content: data.answer }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'bot', content: `Sorry, I encountered an error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app-container">
      {/* Dynamic Background Elements */}
      <div className="bg-glow blob-1"></div>
      <div className="bg-glow blob-2"></div>

      <div className="sidebar">
        <div className="sidebar-header">
          <div className="logo-icon">
            <Sparkles size={24} className="sparkle-icon" />
          </div>
          <div className="logo-text">
            <h1 className="brand-text">Lexa</h1>
            <p className="subtitle">Llama 3 Document Intelligence</p>
          </div>
        </div>

        <div className="section-divider"></div>

        <div className="upload-section">
          <div className="section-title">
            <Database size={16} />
            <h3>Knowledge Base</h3>
          </div>

          <div className="upload-card">
            <div className="file-input-wrapper">
              <input type="file" accept=".pdf,.txt" onChange={handleFileChange} />

              <div className={`upload-drop-area ${file ? 'has-file' : ''}`}>
                {file ? (
                  <div className="file-preview">
                    <div className="file-icon-wrapper">
                      <FileText size={28} />
                    </div>
                    <div className="file-details">
                      <span className="file-name">{file.name}</span>
                      <span className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                    </div>
                    {(!uploadStatus || uploadStatus.type === 'error') && (
                      <button className="remove-file-btn" onClick={removeFile}>
                        <X size={16} />
                      </button>
                    )}
                  </div>
                ) : (
                  <>
                    <div className="upload-icon-pulse">
                      <Upload size={28} />
                    </div>
                    <p className="upload-prompt">Drop a PDF or TXT</p>
                    <p className="upload-subprompt">or click to browse</p>
                  </>
                )}
              </div>
            </div>

            <button
              className={`btn-submit ${file && !uploadStatus ? 'active' : ''}`}
              onClick={handleUpload}
              disabled={!file || uploadStatus?.type === 'loading' || uploadStatus?.type === 'success'}
            >
              {uploadStatus?.type === 'loading' ? (
                <><Loader2 size={18} className="spin" /> Processing Data...</>
              ) : uploadStatus?.type === 'success' ? (
                <><CheckCircle2 size={18} /> Indexed Successfully</>
              ) : (
                'Embed & Index Document'
              )}
            </button>

            {uploadStatus && uploadStatus.type === 'error' && (
              <div className="status-message status-error fade-in">
                <AlertCircle size={16} />
                {uploadStatus.msg}
              </div>
            )}
          </div>

          {uploadedDocs.length > 0 && (
            <div className="doc-list-container fade-in-up">
              <h4 className="doc-list-title">
                <Database size={14} /> Indexed Documents
              </h4>
              <div className="doc-list">
                {uploadedDocs.map((docName, idx) => (
                  <div key={idx} className="doc-item">
                    <FileText size={16} className="doc-item-icon" />
                    <span className="doc-item-name" title={docName}>{docName}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <div className="tech-stack">
            <span>Powered by</span>
            <div className="tech-tags">
              <span className="tag">FastAPI</span>
              <span className="tag">LangChain</span>
              <span className="tag">FAISS</span>
            </div>
          </div>
        </div>
      </div>

      <div className="main-content">
        <div className="top-nav">
          <div className="nav-badge">Secure Local Inference Sandbox</div>
        </div>

        <div className="chat-container">
          <div className="messages-area" ref={chatAreaRef}>
            {messages.map((msg, idx) => (
              <div key={idx} className={`message-wrapper fade-in-up ${msg.role}`}>
                {msg.role === 'bot' && (
                  <div className="avatar bot-avatar">
                    <Bot size={20} />
                  </div>
                )}

                <div className="message-content">
                  {msg.role === 'bot' && <div className="message-sender">Lexa AI</div>}
                  <div className="message-bubble">
                    {msg.content}
                  </div>
                </div>

                {msg.role === 'user' && (
                  <div className="avatar user-avatar">
                    <User size={20} />
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className="message-wrapper bot fade-in">
                <div className="avatar bot-avatar pulse">
                  <Bot size={20} />
                </div>
                <div className="message-content">
                  <div className="message-sender">Lexa AI</div>
                  <div className="message-bubble loading-bubble">
                    <div className="typing-dots">
                      <span></span><span></span><span></span>
                    </div>
                    Retrieving context...
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="input-zone">
            <div className="input-box glass-panel">
              <textarea
                className="chat-textarea"
                placeholder="Ask anything about your documents..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
              />
              <button
                className={`btn-send ${(input.trim() && !loading) ? 'active' : ''}`}
                onClick={handleSend}
                disabled={!input.trim() || loading}
              >
                <Send size={20} className={loading ? "fly" : ""} />
              </button>
            </div>
            <p className="input-disclaimer">AI can make mistakes. Verify important information from the source documents.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
