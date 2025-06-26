import React from 'react';

const Sidebar = ({ 
  sessions, 
  currentSession, 
  onNewSession, 
  onSwitchSession,
  model,
  setModel,
  reportFocus,
  setReportFocus
}) => {
  const models = {
    'deepseek': 'DeepSeek-R1-Distill-Llama-70B',
    'llama': 'Llama-3.3-70B-Versatile',
    'mixtral': 'Mistral-Saba-24B',
    'gemma2': 'Gemma2-9B-IT',
    'qwen': 'Qwen-QWQ-32B',
    'compound-beta-mini': 'Compound-Beta-Mini',
    'gpt-4': 'GPT-4 Turbo',
    'gpt-3.5': 'GPT-3.5 Turbo',
    'claude-3': 'Claude 3 Sonnet',
    'gemini': 'Gemini Pro',
    'mock': 'Mock AI (Demo)'
  };

  const reports = {
    'all': 'All IPCC Reports 🌍',
    'srocc': 'SROCC Summary for Policymakers (2019) 🌊',
    'ar6_syr_full': 'AR6 Synthesis Report Full Volume (2023) 📚',
    'ar6_syr_slides': 'AR6 Synthesis Report Slide Deck (2023) 📽️',
    'ar6_wgii_ts': 'AR6 WGII Technical Summary (2022) 🌿',
    'ar6_wgiii': 'AR6 WGIII Full Report (2022) ⚙️',
    'sr15': 'SR15 1.5°C Full Report (2018) 🔥',
    'srccl': 'SRCCL Full Report (2019) 🌾'
  };

  return (
    <div className="sidebar">
      <h2>Configuration</h2>
      
      <div className="form-group">
        <label>AI Model</label>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {Object.entries(models).map(([key, name]) => (
            <option key={key} value={key}>{name}</option>
          ))}
        </select>
      </div>
      
      <div className="form-group">
        <label>Report Focus</label>
        <select value={reportFocus} onChange={(e) => setReportFocus(e.target.value)}>
          {Object.entries(reports).map(([key, name]) => (
            <option key={key} value={key}>{name}</option>
          ))}
        </select>
      </div>
      
      <div className="divider"></div>
      
      <h3>Session Management</h3>
      <div className="form-group">
        <label>Select Session</label>
        <select 
          value={currentSession} 
          onChange={(e) => onSwitchSession(e.target.value)}
        >
          {sessions.map(session => (
            <option key={session} value={session}>{session}</option>
          ))}
        </select>
      </div>
      
      <div className="button-group">
        <button onClick={onNewSession}>New Session</button>
      </div>
    </div>
  );
};

export default Sidebar;
