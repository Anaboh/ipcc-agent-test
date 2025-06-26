import React from 'react';

const Chat = ({ conversation, onSendMessage, chatEndRef }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim()) {
      onSendMessage(message);
      setMessage('');
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>🌍 IPCC Climate Reports LLM Agent</h1>
        <p>AI-Powered Analysis of Climate Science Reports</p>
      </div>

      <div className="chat-messages">
        {conversation.map((msg, index) => (
          <div 
            key={index} 
            className={`message ${msg.role === 'user' ? 'user-message' : 'assistant-message'}`}
          >
            <strong>{msg.role === 'user' ? 'You:' : 'Assistant:'}</strong>
            <div dangerouslySetInnerHTML={{ __html: msg.content.replace(/\n/g, '<br>') }} />
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="chat-input">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Ask about IPCC reports or search recent data..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default Chat;
