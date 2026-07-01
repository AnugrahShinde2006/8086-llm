import { useState, useEffect, useRef } from 'react';
import { Send, Cpu, Layers, Activity } from 'lucide-react';
import './index.css';

// Syntax highlighting utility for 8086 Assembly
const highlightAssembly = (code) => {
  const mnemonics = ['MOV', 'ADD', 'SUB', 'MUL', 'DIV', 'INC', 'DEC', 'CMP', 'AND', 'OR', 'XOR', 'JMP', 'CALL', 'RET', 'HLT', 'LOOP', 'PUSH', 'POP', 'XCHG'];
  const registers = ['AX', 'BX', 'CX', 'DX', 'AH', 'AL', 'BH', 'BL', 'CH', 'CL', 'DH', 'DL', 'CS', 'DS', 'SS', 'ES', 'SP', 'BP', 'SI', 'DI'];
  
  let html = code;
  // This is a naive regex replacement for demonstration.
  mnemonics.forEach(m => {
    const reg = new RegExp(`\\b${m}\\b`, 'g');
    html = html.replace(reg, `<span class="asm-keyword">${m}</span>`);
  });
  registers.forEach(r => {
    const reg = new RegExp(`\\b${r}\\b`, 'g');
    html = html.replace(reg, `<span class="asm-register">${r}</span>`);
  });
  html = html.replace(/\b([0-9A-Fa-f]+H)\b/g, '<span class="asm-hex">$1</span>');
  html = html.replace(/(;.*)/g, '<span class="asm-comment">$1</span>');
  
  return html;
};

export default function App() {
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    fetch('http://localhost:8000/stats')
      .then(res => res.json())
      .then(data => setStats(data))
      .catch(console.error);
  }, []);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async () => {
    if (!prompt.trim()) return;

    const userMsg = { role: 'user', content: prompt };
    setMessages(prev => [...prev, userMsg]);
    setPrompt('');
    setLoading(true);

    try {
      const res = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: userMsg.content, max_tokens: 1024 })
      });
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.reply }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error connecting to the backend. Ensure FastAPI is running.' }]);
    } finally {
      setLoading(false);
    }
  };

  const renderContent = (content) => {
    // If it looks like assembly code
    if (content.includes('MOV') || content.includes('AX') || content.includes('HLT') || content.includes(';')) {
      return (
        <div className="asm-code" dangerouslySetInnerHTML={{ __html: highlightAssembly(content) }} />
      );
    }
    return content;
  };

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <Cpu size={24} />
          8086 LLM
        </div>
        
        {stats && (
          <div className="stats-card">
            <h3>Model Stats</h3>
            <div className="stat-row">
              <span>Model</span>
              <span className="stat-value">{stats.model}</span>
            </div>
            <div className="stat-row">
              <span>Type</span>
              <span className="stat-value">{stats.type}</span>
            </div>
            <div className="stat-row" style={{ marginTop: '1rem', borderTop: '1px solid var(--border-color)', paddingTop: '0.5rem' }}>
              <span>Device</span>
              <span className="stat-value" style={{ color: '#34d399' }}>{stats.device}</span>
            </div>
          </div>
        )}
      </aside>

      <main className="main-chat-area">
        <div className="chat-history">
          {messages.length === 0 && (
            <div style={{ margin: 'auto', textAlign: 'center', color: 'var(--text-muted)' }}>
              <Layers size={48} style={{ margin: '0 auto 1rem auto', opacity: 0.5 }} />
              <h2>Domain-Specific AI for Intel 8086</h2>
              <p>Ask a question or request an assembly program.</p>
            </div>
          )}
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-content">
                {msg.role === 'assistant' ? renderContent(msg.content) : msg.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <div className="message-content" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Activity size={16} className="spinner" /> Generating...
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="input-area">
          <div className="input-container">
            <input 
              type="text" 
              className="chat-input"
              placeholder="Write an 8086 assembly program to add two numbers..."
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSend()}
            />
            <button className="send-btn" onClick={handleSend} disabled={!prompt.trim() || loading}>
              <Send size={18} />
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
