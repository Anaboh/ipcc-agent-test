from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import datetime
import re
import random
import time
import numpy as np
import pickle
import asyncio
from typing import List, Dict, Tuple, Optional
import pandas as pd
import nest_asyncio
nest_asyncio.apply()

# [Include all the original Python code here from Cecil-version.py, 
# but replace the Streamlit parts with Flask API endpoints]

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

agent = IPCCLLMAgent()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']
    model = data.get('model', 'mock')
    report_focus = data.get('report_focus', 'all')
    
    # Process message using agent
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    history, _ = loop.run_until_complete(
        agent.process_message(message, agent.conversation_history, model, report_focus)
    )
    
    return jsonify({
        'conversation': history,
        'last_response': history[-1]['content'] if history else ""
    })

@app.route('/api/new_session', methods=['POST'])
def new_session():
    agent.new_session()
    return jsonify({
        'session_id': agent.session_id,
        'conversation': agent.conversation_history
    })

@app.route('/api/switch_session', methods=['POST'])
def switch_session():
    data = request.json
    session_id = data['session_id']
    agent.switch_session(session_id)
    return jsonify({
        'session_id': agent.session_id,
        'conversation': agent.conversation_history
    })

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    return jsonify(agent.get_session_list())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
