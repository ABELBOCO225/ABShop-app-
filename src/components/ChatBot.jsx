import { useState, useRef, useEffect } from 'react';
import './ChatBot.css';

const SYSTEM_PROMPT = `Tu es l'assistant virtuel d'AbShop, une boutique en ligne algérienne.
Tu réponds uniquement en français, de façon naturelle et chaleureuse.
Tu es serviable, concis (2-3 phrases max) et toujours poli.

Informations sur AbShop :
- Produits : électronique, vêtements, accessoires
- Livraison : partout en Algérie, 3 à 5 jours ouvrables
- Livraison gratuite à partir de 5000 DA
- Paiement : carte bancaire, virement, paiement à la livraison
- Retours : 14 jours après réception
- Remboursement : sous 5 jours après retour reçu
- Support : contact@abshop.com, réponse sous 24h
- Promos : -10% sur la première commande via newsletter
- Suivi commande : numéro de suivi envoyé par email après expédition

Règles :
- Si la question ne concerne pas AbShop, réponds poliment.
- Ne jamais inventer de prix spécifiques.
- Toujours proposer contact@abshop.com pour les cas complexes.`;

const GEMINI_KEY = import.meta.env.VITE_GEMINI_API_KEY;
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key=${GEMINI_KEY}`;

export default function ChatBot() {
  const [open,     setOpen]     = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Bonjour ! Je suis l\'assistant AbShop. Comment puis-je vous aider ? 😊' }
  ]);
  const [input,   setInput]   = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function sendMessage() {
    const text = input.trim();
    if (!text || loading) return;

    const newMessages = [...messages, { role: 'user', content: text }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const contents = [
        { role: 'user',  parts: [{ text: SYSTEM_PROMPT }] },
        { role: 'model', parts: [{ text: 'Compris ! Je suis l\'assistant AbShop.' }] },
        ...messages.slice(-10).map(m => ({
          role:  m.role === 'assistant' ? 'model' : 'user',
          parts: [{ text: m.content }]
        })),
        { role: 'user', parts: [{ text }] }
      ];

      const res  = await fetch(GEMINI_URL, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          contents,
          generationConfig: { maxOutputTokens: 300, temperature: 0.7 }
        })
      });

      const data  = await res.json();
      const reply = data.candidates?.[0]?.content?.parts?.[0]?.text || 'Pas de réponse.';
      setMessages(prev => [...prev, { role: 'assistant', content: reply }]);

    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Désolé, une erreur est survenue. Réessayez.'
      }]);
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <>
      <button className="chatbot-toggle" onClick={() => setOpen(o => !o)}>
        {open ? '✕' : '💬'}
      </button>

      {open && (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <span className="chatbot-avatar">🛍️</span>
            <div>
              <div className="chatbot-name">AbShop Assistant</div>
              <div className="chatbot-status">● En ligne</div>
            </div>
          </div>

          <div className="chatbot-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`chatbot-msg ${msg.role}`}>
                {msg.role === 'assistant' && <span className="chatbot-msg-avatar">🛍️</span>}
                <div className="chatbot-bubble">{msg.content}</div>
              </div>
            ))}
            {loading && (
              <div className="chatbot-msg assistant">
                <span className="chatbot-msg-avatar">🛍️</span>
                <div className="chatbot-bubble chatbot-typing">
                  <span></span><span></span><span></span>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="chatbot-input-bar">
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Votre message..."
              disabled={loading}
            />
            <button onClick={sendMessage} disabled={loading || !input.trim()}>➤</button>
          </div>
        </div>
      )}
    </>
  );
}
