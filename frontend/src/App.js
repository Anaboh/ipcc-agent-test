import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import Chat from './components/Chat';
import './App.css';

function App() {
  const [conversation, setConversation] = useState([]);
  const [sessionId, setSessionId] = useState('');
  const [sessions, setSessions] = useState([]);
  const [model, setModel] = useState('mock');
  const [reportFocus, setReportFocus] = useState('all');
  const chatEndRef = useRef(null);

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    const response = await fetch('/api/sessions');
    const data = await response.json();
    setSessions(data);
    if (data.length > 0) {
      setSessionId(data[0]);
    }
  };

  const handleNewSession = async () => {
    const response = await fetch('/api/new_session', { method: 'POST' });
    const data = await response.json();
    setConversation(data.conversation);
    setSessionId(data.session_id);
    fetchSessions();
  };

  const handleSwitchSession = async (sessionId) => {
    const response = await fetch('/api/switch_session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    });
    const data = await response.json();
    setConversation(data.conversation);
    setSessionId(data.session_id);
  };

  const handleSendMessage = async (message) => {
    // Add user message immediately
    const userMessage = { role: "user", content: message };
    const newConversation = [...conversation, userMessage];
    setConversation(newConversation);

    // Send to backend
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        message, 
        model,
        report_focus: reportFocus 
      })
    });
    
    const data = await response.json();
    setConversation(data.conversation);
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation]);

  return (
    <div className="app">
      <Sidebar 
        sessions={sessions}
        currentSession={sessionId}
        onNewSession={handleNewSession}
        onSwitchSession={handleSwitchSession}
        model={model}
        setModel={setModel}
        reportFocus={reportFocus}
        setReportFocus={setReportFocus}
      />
      
      <Chat 
        conversation={conversation}
        onSendMessage={handleSendMessage}
        chatEndRef={chatEndRef}
      />
    </div>
  );
}

export default App;
