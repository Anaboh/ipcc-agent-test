# -*- coding: utf-8 -*-
import streamlit as st
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

# Try imports with fallbacks
try:
    import faiss
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    # Updated import path
    from langchain_community.embeddings import HuggingFaceEmbeddings
    FAISS_AVAILABLE = True
except ImportError as e:
    FAISS_AVAILABLE = False
    print(f"FAISS import error: {str(e)}")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Custom CSS for dyslexia-friendly and theme support
DYSLEXIA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=OpenDyslexic:wght@400;700&display=swap');

* {
    font-family: 'OpenDyslexic', sans-serif;
    letter-spacing: 0.05em;
    line-height: 1.6;
}

/* Light mode */
[data-theme="light"] {
    --primary: #f0f2f6;
    --secondary: #ffffff;
    --text: #31333F;
    --accent: #4e73df;
    --card: #ffffff;
    --border: #e0e0e0;
}

/* Dark mode */
[data-theme="dark"] {
    --primary: #1e1e2e;
    --secondary: #252535;
    --text: #f0f0f0;
    --accent: #667eea;
    --card: #2d2d3d;
    --border: #444455;
}

body {
    background-color: var(--primary);
    color: var(--text);
    transition: all 0.3s;
}

.stApp {
    background-color: var(--primary) !important;
}

.stChatInput {
    background-color: var(--secondary) !important;
    border: 1px solid var(--border) !important;
}

.stButton button {
    background-color: var(--accent) !important;
    color: white !important;
    border-radius: 20px !important;
}

.stSelectbox, .stTextInput {
    background-color: var(--secondary) !important;
    border: 1px solid var(--border) !important;
}

.chat-message {
    padding: 15px;
    margin: 10px 0;
    border-radius: 15px;
    max-width: 85%;
}

.user-message {
    background-color: var(--accent);
    color: white;
    margin-left: auto;
}

.assistant-message {
    background-color: var(--card);
    border: 1px solid var(--border);
}

.source-badge {
    background-color: var(--accent);
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.8em;
    margin-top: 5px;
    display: inline-block;
}

.theme-toggle {
    position: fixed;
    top: 15px;
    right: 15px;
    z-index: 999;
}

.container {
    max-width: 1000px;
    margin: 0 auto;
    padding: 20px;
}

@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
    .chat-message {
        max-width: 90%;
    }
}
</style>
"""

class FAISSRetriever(BaseRetriever):
    def __init__(self, report_key: str, index_path: str, texts_path: str, embed_model: HuggingFaceEmbeddings, k: int = 5):
        super().__init__()
        self._report_key = report_key
        self._embed_model = embed_model
        self._index = faiss.read_index(index_path)
        with open(texts_path, 'rb') as f:
            self._texts = pickle.load(f)
        self._k = k

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            query_embedding = self._embed_model.embed_query(query)
            query_embedding = np.array([query_embedding], dtype=np.float32)
            distances, indices = self._index.search(query_embedding, self._k * 20)
            relevant_docs = []
            for i, distance in zip(indices[0], distances[0]):
                if len(relevant_docs) >= self._k:
                    break
                if distance > 0.65:
                    continue
                text_entry = self._texts[i]
                text = text_entry['text'] if isinstance(text_entry, dict) else str(text_entry)
                metadata = text_entry.get('metadata', {}) if isinstance(text_entry, dict) else {}
                report_key = metadata.get('report_key', 'Unknown')
                if report_key == self._report_key:
                    relevant_docs.append(Document(page_content=text, metadata=metadata))
            return relevant_docs
        except Exception as e:
            print(f"Error retrieving from {self._report_key} index: {str(e)}")
            return []

class IPCCLLMAgent:
    def __init__(self):
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_dir = "./history/"
        self.conversation_history = self.load_conversation_history()
        self.ipcc_reports = {
            'all': {'name': 'All IPCC Reports', 'color': '🌍'},
            'srocc': {'name': 'SROCC Summary for Policymakers (2019)', 'color': '🌊'},
            'ar6_syr_full': {'name': 'AR6 Synthesis Report Full Volume (2023)', 'color': '📚'},
            'ar6_syr_slides': {'name': 'AR6 Synthesis Report Slide Deck (2023)', 'color': '📽️'},
            'ar6_wgii_ts': {'name': 'AR6 WGII Technical Summary (2022)', 'color': '🌿'},
            'ar6_wgiii': {'name': 'AR6 WGIII Full Report (2022)', 'color': '⚙️'},
            'sr15': {'name': 'SR15 1.5°C Full Report (2018)', 'color': '🔥'},
            'srccl': {'name': 'SRCCL Full Report (2019)', 'color': '🌾'},
        }
        self.llm_models = {
            'deepseek': {'name': 'DeepSeek-R1-Distill-Llama-70B', 'provider': 'Groq'},
            'llama': {'name': 'Llama-3.3-70B-Versatile', 'provider': 'Groq'},
            'mixtral': {'name': 'Mistral-Saba-24B', 'provider': 'Groq'},
            'gemma2': {'name': 'Gemma2-9B-IT', 'provider': 'Groq'},
            'qwen': {'name': 'Qwen-QWQ-32B', 'provider': 'Groq'},
            'compound-beta-mini': {'name': 'Compound-Beta-Mini', 'provider': 'Groq'},
            'gpt-4': {'name': 'GPT-4 Turbo', 'provider': 'OpenAI'},
            'gpt-3.5': {'name': 'GPT-3.5 Turbo', 'provider': 'OpenAI'},
            'claude-3': {'name': 'Claude 3 Sonnet', 'provider': 'Anthropic'},
            'gemini': {'name': 'Gemini Pro', 'provider': 'Google'},
            'mock': {'name': 'Mock AI (Demo)', 'provider': 'Local'}
        }
        self.setup_api_clients()
        self.ipcc_knowledge = self.load_ipcc_knowledge()
        
        # Temporarily disable FAISS retrievers
        self.faiss_retrievers = {}
        # self.faiss_retrievers = self.setup_faiss_retrievers()

    def setup_api_clients(self):
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_client = None
        self.groq_client_llama = None
        self.groq_client_deepseek = None
        self.groq_client_mixtral = None
        self.groq_client_gemma2 = None
        self.groq_client_qwen = None
        self.groq_client_compound_beta_mini = None

        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        gemini_key = os.getenv('GEMINI_API_KEY')
        groq_key = os.getenv('GROQ_API_KEY')

        if openai_key and OPENAI_AVAILABLE:
            self.openai_client = openai.OpenAI(api_key=openai_key)

        if anthropic_key and ANTHROPIC_AVAILABLE:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)

        if gemini_key and GEMINI_AVAILABLE:
            genai.configure(api_key=gemini_key)
            self.gemini_client = genai.GenerativeModel('gemini-pro')

        if groq_key and GROQ_AVAILABLE:
            try:
                self.groq_client_llama = ChatGroq(api_key=groq_key, model_name='llama-3.3-70b-versatile')
                try:
                    self.groq_client_deepseek = ChatGroq(api_key=groq_key, model_name='deepseek-r1-distill-llama-70b')
                except Exception:
                    self.groq_client_deepseek = ChatGroq(api_key=groq_key, model_name='deepseek-r1')
                self.groq_client_mixtral = ChatGroq(api_key=groq_key, model_name='mistral-saba-24b')
                self.groq_client_gemma2 = ChatGroq(api_key=groq_key, model_name='gemma2-9b-it')
                self.groq_client_qwen = ChatGroq(api_key=groq_key, model_name='qwen-qwq-32b')
                self.groq_client_compound_beta_mini = ChatGroq(api_key=groq_key, model_name='compound-beta-mini')
            except Exception as e:
                print(f"Error initializing Groq clients: {str(e)}")

    def setup_faiss_retrievers(self) -> Dict[str, Optional[FAISSRetriever]]:
        if not FAISS_AVAILABLE:
            return {}

        base_path = "./faiss_data/"
        retrievers = {}

        file_mapping = {
            'srocc': {
                'index': '01_SROCC_SPM_FINAL_index.bin',
                'texts': '01_SROCC_SPM_FINAL_texts.pkl'
            },
            'ar6_syr_full': {
                'index': 'IPCC_AR6_SYR_FullVolume_index.bin',
                'texts': 'IPCC_AR6_SYR_FullVolume_texts.pkl'
            },
            'ar6_syr_slides': {
                'index': 'IPCC_AR6_SYR_SlideDeck_index.bin',
                'texts': 'IPCC_AR6_SYR_SlideDeck_texts.pkl'
            },
            'ar6_wgii_ts': {
                'index': 'IPCC_AR6_WGII_TechnicalSummary_index.bin',
                'texts': 'IPCC_AR6_WGII_TechnicalSummary_texts.pkl'
            },
            'ar6_wgiii': {
                'index': 'IPCC_AR6_WGIII_FullReport_index.bin',
                'texts': 'IPCC_AR6_WGIII_FullReport_texts.pkl'
            },
            'sr15': {
                'index': 'SR15_Full_Report_LR_index.bin',
                'texts': 'SR15_Full_Report_LR_texts.pkl'
            },
            'srccl': {
                'index': 'SRCCL_Full_Report_index.bin',
                'texts': 'SRCCL_Full_Report_texts.pkl'
            }
        }

        try:
            embed_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
            for report_key in self.ipcc_reports.keys():
                if report_key == 'all':
                    continue
                if report_key not in file_mapping:
                    continue
                index_path = os.path.join(base_path, file_mapping[report_key]['index'])
                texts_path = os.path.join(base_path, file_mapping[report_key]['texts'])
                if os.path.exists(index_path) and os.path.exists(texts_path):
                    try:
                        retriever = FAISSRetriever(report_key, index_path, texts_path, embed_model, k=5)
                        retrievers[report_key] = retriever
                    except Exception:
                        pass
            return retrievers
        except Exception as e:
            print(f"Error setting up FAISS retrievers: {str(e)}")
            return {}

    def load_ipcc_knowledge(self) -> Dict:
        return {
            'srocc_summary': {
                'content': """# 🌊 SROCC Summary for Policymakers: Key Findings
- **Ocean Warming**: Oceans have absorbed 90% of excess heat since 1970.
- **Sea Level Rise**: Global mean sea level rising at 3.7 mm/year, accelerating.
- **Glacier and Ice Loss**: Arctic sea ice declining; Greenland and Antarctic ice sheets losing mass.
- **Marine Ecosystems**: Coral bleaching and fishery declines due to warming and acidification.
- **Coastal Risks**: Increased flooding and erosion affecting millions by 2100.
- **Adaptation Needs**: Enhanced coastal defenses and ecosystem restoration.""",
                'sources': ['SROCC SPM (2019)']
            },
            'sr15_summary': {
                'content': """# 🔥 SR15 1.5°C Full Report: Key Findings
- **1.5°C vs. 2°C**: Half a degree reduces severe impacts significantly.
- **Carbon Budget**: ~420 GtCO₂ remaining for 1.5°C (50% chance, 2018).
- **Emission Cuts**: 45% reduction by 2030, net zero by 2050.
- **Impacts**: Lower risks to ecosystems, health, and food security at 1.5°C.
- **Solutions**: Rapid energy transition, reforestation, and carbon capture.""",
                'sources': ['SR15 Full Report (2018)']
            },
            'srccl_summary': {
                'content': """# 🌾 SRCCL Full Report: Key Findings
- **Land Degradation**: 23% of global land degraded, reducing carbon sinks.
- **Food Security**: Climate change exacerbates hunger; 821 million undernourished.
- **Deforestation**: Contributes 11% of GHG emissions.
- **Solutions**: Sustainable land management, dietary shifts, and reforestation.
- **Co-benefits**: Improved biodiversity, soil health, and livelihoods.""",
                'sources': ['SRCCL Full Report (2019)']
            },
            'ar6_wgii_summary': {
                'content': """# 🌿 AR6 WGII Technical Summary: Key Findings
- **Vulnerable Populations**: 3.3–3.6 billion people in high-risk areas.
- **Ecosystem Impacts**: 14% of species at high extinction risk at 1.5°C.
- **Health Risks**: Increased heat-related mortality and disease spread.
- **Adaptation Gaps**: Current measures insufficient for 2°C scenarios.
- **Solutions**: Climate-resilient development and ecosystem-based adaptation.""",
                'sources': ['AR6 WGII Technical Summary (2022)']
            },
            'ar6_wgiii_summary': {
                'content': """# ⚙️ AR6 WGIII Full Report: Key Findings
- **Emission Trends**: GHG emissions rose 54% from 1990 to 2019.
- **1.5°C Pathway**: Peak emissions by 2025, 43% cut by 2030.
- **Sector Solutions**: Renewables, electrification, and efficiency improvements.
- **Costs**: Net-zero by 2050 achievable with 2–3% GDP investment.
- **Policy Needs**: Carbon pricing, subsidies reform, and just transitions.""",
                'sources': ['AR6 WGIII Full Report (2022)']
            }
        }

    def load_conversation_history(self) -> List[Dict]:
        os.makedirs(self.history_dir, exist_ok=True)
        history_file = os.path.join(self.history_dir, f"history_{self.session_id}.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def save_conversation_history(self):
        history_file = os.path.join(self.history_dir, f"history_{self.session_id}.json")
        try:
            with open(history_file, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception:
            pass

    def get_session_list(self) -> List[str]:
        os.makedirs(self.history_dir, exist_ok=True)
        sessions = []
        for filename in os.listdir(self.history_dir):
            if filename.startswith("history_") and filename.endswith(".json"):
                session_id = filename.replace("history_", "").replace(".json", "")
                sessions.append(session_id)
        return sorted(sessions, reverse=True)

    def switch_session(self, session_id: str) -> List[Dict]:
        self.session_id = session_id
        self.conversation_history = self.load_conversation_history()
        return self.conversation_history

    def new_session(self) -> List[Dict]:
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_history = []
        self.save_conversation_history()
        return self.conversation_history

    def format_response(self, content: str, sources: List[str] = None, report_focus: str = 'all') -> str:
        formatted = f"**Report Focus**: {self.ipcc_reports[report_focus]['name']}\n\n{content}"
        if sources:
            formatted += f"\n\n**Sources**: {', '.join(sources)}"
        return formatted

    def get_mock_response(self, message: str, report_focus: str) -> Tuple[str, List[str]]:
        message_lower = message.lower()
        if report_focus == 'srocc' or 'srocc' in message_lower or 'ocean' in message_lower:
            knowledge = self.ipcc_knowledge['srocc_summary']
            return knowledge['content'], knowledge['sources']
        elif report_focus == 'sr15' or '1.5' in message_lower or 'sr15' in message_lower:
            knowledge = self.ipcc_knowledge['sr15_summary']
            return knowledge['content'], knowledge['sources']
        elif report_focus == 'srccl' or 'land' in message_lower or 'srccl' in message_lower:
            knowledge = self.ipcc_knowledge['srccl_summary']
            return knowledge['content'], knowledge['sources']
        elif report_focus == 'ar6_wgii_ts' or 'impacts' in message_lower or 'wgii' in message_lower:
            knowledge = self.ipcc_knowledge['ar6_wgii_summary']
            return knowledge['content'], knowledge['sources']
        elif report_focus == 'ar6_wgiii' or 'mitigation' in message_lower or 'wgiii' in message_lower:
            knowledge = self.ipcc_knowledge['ar6_wgiii_summary']
            return knowledge['content'], knowledge['sources']
        elif report_focus in ['ar6_syr_full', 'ar6_syr_slides'] or 'synthesis' in message_lower:
            return ("Placeholder: AR6 Synthesis Report (Full or Slides) summary not available in mock mode."), ['Mock Response']
        else:
            return """I can help with IPCC reports! Try asking about:
- SROCC ocean and cryosphere findings
- SR15 1.5°C pathways
- SRCCL land use impacts
- AR6 WGII impacts and adaptation
- AR6 WGIII mitigation strategies""", ['IPCC Knowledge Base']

    async def clean_response(self, content: str) -> str:
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        return cleaned.strip()

    def requires_web_search(self, message: str) -> bool:
        message_lower = message.lower()
        web_search_keywords = [
            'search', 'web', 'internet', 'online', 'recent', 'latest', 'current',
            'news', 'update', 'real-time', 'live data', 'fetch'
        ]
        return any(keyword in message_lower for keyword in web_search_keywords)

    async def call_llm_api(self, messages: List[Dict], model: str, report_focus: str) -> Tuple[str, List[str]]:
        if model == 'mock':
            time.sleep(1)
            user_message = messages[-1]['content']
            return self.get_mock_response(user_message, report_focus)

        # ... (rest of call_llm_api method remains the same as original) ...

    async def process_message(self, message: str, history: List[Dict], model: str, report_focus: str) -> Tuple[List[Dict], str]:
        if not message.strip():
            return history, ""
        self.conversation_history.append({"role": "user", "content": message})
        messages = self.conversation_history.copy()
        try:
            content, sources = await self.call_llm_api(messages, model, report_focus)
            formatted_response = self.format_response(content, sources, report_focus)
            self.conversation_history.append({"role": "assistant", "content": formatted_response})
            self.save_conversation_history()
        except Exception as e:
            error_response = f"⚠️ Error: {str(e)}.\n\nPlease try again."
            self.conversation_history.append({"role": "assistant", "content": error_response})
            self.save_conversation_history()
        return self.conversation_history, ""

def main():
    # Initialize agent
    if 'agent' not in st.session_state:
        st.session_state.agent = IPCCLLMAgent()
    
    # Set theme (default to light)
    if 'theme' not in st.session_state:
        st.session_state.theme = "light"
    
    # Apply CSS
    st.markdown(DYSLEXIA_CSS, unsafe_allow_html=True)
    st.markdown(f'<body data-theme="{st.session_state.theme}">', unsafe_allow_html=True)
    
    # Theme toggle
    if st.session_state.theme == "light":
        toggle_label = "🌙 Dark Mode"
    else:
        toggle_label = "☀️ Light Mode"
    
    if st.button(toggle_label, key="theme-toggle", use_container_width=False):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()
    
    # App header
    st.title("🌍 IPCC Climate Reports LLM Agent")
    st.caption("AI-Powered Analysis of Climate Science Reports")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        model_option = st.selectbox(
            "AI Model",
            options=list(st.session_state.agent.llm_models.keys()),
            format_func=lambda x: st.session_state.agent.llm_models[x]['name'],
            index=0
        )
        
        report_option = st.selectbox(
            "Report Focus",
            options=list(st.session_state.agent.ipcc_reports.keys()),
            format_func=lambda x: st.session_state.agent.ipcc_reports[x]['name'],
            index=0
        )
        
        st.divider()
        st.subheader("Session Management")
        session_options = st.session_state.agent.get_session_list()
        session_selection = st.selectbox(
            "Select Session",
            options=session_options,
            index=0
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Switch Session", use_container_width=True):
                st.session_state.agent.switch_session(session_selection)
                st.rerun()
                
        with col2:
            if st.button("New Session", use_container_width=True):
                st.session_state.agent.new_session()
                st.rerun()
                
        st.divider()
        st.subheader("System Status")
        groq_status = "✅ Available" if any([
            st.session_state.agent.groq_client_llama,
            st.session_state.agent.groq_client_deepseek,
            st.session_state.agent.groq_client_mixtral,
            st.session_state.agent.groq_client_gemma2,
            st.session_state.agent.groq_client_qwen,
            st.session_state.agent.groq_client_compound_beta_mini
        ]) else "❌ Unavailable"
        
        st.write(f"Groq: {groq_status}")
        st.write(f"OpenAI: {'✅ Available' if st.session_state.agent.openai_client else '❌ Unavailable'}")
        st.write(f"Anthropic: {'✅ Available' if st.session_state.agent.anthropic_client else '❌ Unavailable'}")
        st.write(f"Gemini: {'✅ Available' if st.session_state.agent.gemini_client else '❌ Unavailable'}")
        st.write(f"FAISS: {'✅ Loaded' if st.session_state.agent.faiss_retrievers else '❌ Not Loaded'}")
        st.write("✅ Mock AI: Always Available")
    
    # Chat display
    st.subheader("Climate Science Chat")
    for message in st.session_state.agent.conversation_history:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{content}</div>', 
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong><br>{content}</div>', 
                        unsafe_allow_html=True)
    
    # User input
    if prompt := st.chat_input("Ask about IPCC reports or search recent data..."):
        # Display user message immediately
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{prompt}</div>', 
                    unsafe_allow_html=True)
        
        # Generate and display assistant response
        with st.spinner("Analyzing climate data..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            history, _ = loop.run_until_complete(
                st.session_state.agent.process_message(
                    prompt, 
                    st.session_state.agent.conversation_history, 
                    model_option, 
                    report_option
                )
            )
            # Get the last assistant message
            last_assistant_msg = st.session_state.agent.conversation_history[-1]["content"]
            st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong><br>{last_assistant_msg}</div>', 
                        unsafe_allow_html=True)
    
    # Quick prompts
    st.divider()
    st.subheader("💡 Quick Prompts")
    quick_prompts = [
        "Summarize SROCC key findings",
        "Key points from SR15 1.5°C report",
        "Land use impacts from SRCCL",
        "Summarize AR6 WGIII mitigation strategies",
        "Explain AR6 WGII impacts"
    ]
    
    cols = st.columns(len(quick_prompts))
    for i, col in enumerate(cols):
        with col:
            if st.button(quick_prompts[i], use_container_width=True):
                # Trigger the chat input with the prompt
                st.session_state.prompt = quick_prompts[i]
                st.rerun()

if __name__ == "__main__":
    main()
