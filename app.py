import os
import streamlit as st
import google.generativeai as genai
import yaml
from pathlib import Path
import pandas as pd
import io
import json

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Multi-Agent Analysis Hub", page_icon="🌍", layout="wide")

# -------------------- THEME DEFINITIONS --------------------
THEMES = {
    "Iceland": {
        "primary": "#00D9FF",
        "secondary": "#B8E6F0",
        "bg_main": "#0A1929",
        "bg_secondary": "#132F4C",
        "text": "#E3F2FD",
        "accent": "#82E9FF",
        "icon": "🇮🇸",
        "gradient": "linear-gradient(135deg, #0A1929 0%, #1A3A52 100%)"
    },
    "Canada": {
        "primary": "#FF0000",
        "secondary": "#FFB3B3",
        "bg_main": "#1C0000",
        "bg_secondary": "#330000",
        "text": "#FFFFFF",
        "accent": "#FF6B6B",
        "icon": "🇨🇦",
        "gradient": "linear-gradient(135deg, #1C0000 0%, #4D0000 100%)"
    },
    "Paris": {
        "primary": "#FFD700",
        "secondary": "#FFF4D6",
        "bg_main": "#1A1A2E",
        "bg_secondary": "#16213E",
        "text": "#F0E68C",
        "accent": "#DAA520",
        "icon": "🇫🇷",
        "gradient": "linear-gradient(135deg, #1A1A2E 0%, #2D3561 100%)"
    },
    "Rome": {
        "primary": "#C19A6B",
        "secondary": "#E8D5B7",
        "bg_main": "#2C1810",
        "bg_secondary": "#3D2414",
        "text": "#F5E6D3",
        "accent": "#D4AF7A",
        "icon": "🇮🇹",
        "gradient": "linear-gradient(135deg, #2C1810 0%, #5C3A24 100%)"
    },
    "Venice": {
        "primary": "#4A90E2",
        "secondary": "#A7D8FF",
        "bg_main": "#0F2027",
        "bg_secondary": "#203A43",
        "text": "#E0F7FF",
        "accent": "#6BB6FF",
        "icon": "🛶",
        "gradient": "linear-gradient(135deg, #0F2027 0%, #2C5364 100%)"
    },
    "Copenhagen": {
        "primary": "#FF6B9D",
        "secondary": "#FFD1DC",
        "bg_main": "#1A0B1A",
        "bg_secondary": "#2D1F2D",
        "text": "#FFE4E9",
        "accent": "#FF8FB3",
        "icon": "🇩🇰",
        "gradient": "linear-gradient(135deg, #1A0B1A 0%, #3D2E3D 100%)"
    },
    "Munich": {
        "primary": "#00A8E8",
        "secondary": "#B3E5FC",
        "bg_main": "#001F3F",
        "bg_secondary": "#003B5C",
        "text": "#E1F5FE",
        "accent": "#4FC3F7",
        "icon": "🇩🇪",
        "gradient": "linear-gradient(135deg, #001F3F 0%, #005073 100%)"
    },
    "Swiss": {
        "primary": "#E74C3C",
        "secondary": "#FADBD8",
        "bg_main": "#0E0E0E",
        "bg_secondary": "#1C1C1C",
        "text": "#FFFFFF",
        "accent": "#EC7063",
        "icon": "🇨🇭",
        "gradient": "linear-gradient(135deg, #0E0E0E 0%, #2D2D2D 100%)"
    },
    "Space": {
        "primary": "#9D00FF",
        "secondary": "#E0B3FF",
        "bg_main": "#000000",
        "bg_secondary": "#0D0221",
        "text": "#E0E0E0",
        "accent": "#BD00FF",
        "icon": "🚀",
        "gradient": "linear-gradient(135deg, #000000 0%, #1a0033 50%, #0D0221 100%)"
    }
}

# -------------------- SESSION STATE INITIALIZATION --------------------
if 'theme' not in st.session_state:
    st.session_state.theme = "Space"
if 'workflow_data' not in st.session_state:
    st.session_state.workflow_data = {}
if 'GEMINI_API_KEY' not in st.session_state:
    st.session_state['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY", "")
if 'num_datasets' not in st.session_state:
    st.session_state.num_datasets = 1
if 'datasets' not in st.session_state:
    st.session_state.datasets = []
if 'datasets_json' not in st.session_state:
    st.session_state.datasets_json = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'config'

# -------------------- DYNAMIC CSS --------------------
def apply_theme(theme_name):
    theme = THEMES[theme_name]
    st.markdown(f"""
        <style>
            .main {{
                background: {theme['gradient']};
                color: {theme['text']};
            }}
            h1, h2, h3 {{
                color: {theme['primary']};
                text-align: center;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }}
            .stButton>button {{
                background: linear-gradient(135deg, {theme['primary']} 0%, {theme['accent']} 100%);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: bold;
                padding: 12px 24px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }}
            .stButton>button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            }}
            .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {{
                background-color: {theme['bg_secondary']};
                color: {theme['text']};
                border: 1px solid {theme['primary']};
                border-radius: 8px;
            }}
            .stFileUploader>div {{
                border: 2px dashed {theme['primary']};
                background-color: {theme['bg_secondary']};
                border-radius: 10px;
            }}
            .stExpander {{
                background-color: {theme['bg_secondary']};
                border: 1px solid {theme['primary']};
                border-radius: 10px;
            }}
            .keyword {{
                color: {theme['accent']};
                font-weight: bold;
                text-shadow: 0 0 5px {theme['accent']};
            }}
            .results-box, .markdown-preview, .json-preview {{
                background-color: {theme['bg_secondary']};
                border: 1px solid {theme['primary']};
                padding: 20px;
                border-radius: 12px;
                margin-top: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            .dataset-card {{
                background-color: {theme['bg_secondary']};
                border: 2px solid {theme['primary']};
                border-radius: 12px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            .step-indicator {{
                background: linear-gradient(135deg, {theme['primary']} 0%, {theme['accent']} 100%);
                color: white;
                padding: 10px 20px;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                margin: 20px 0;
            }}
            [data-testid="stSidebar"] {{
                background: {theme['gradient']};
            }}
            .stRadio > div {{
                background-color: {theme['bg_secondary']};
                padding: 10px;
                border-radius: 8px;
            }}
            div[data-baseweb="select"] > div {{
                background-color: {theme['bg_secondary']};
                border-color: {theme['primary']};
            }}
            .stNumberInput>div>div>input {{
                background-color: {theme['bg_secondary']};
                color: {theme['text']};
                border: 1px solid {theme['primary']};
            }}
        </style>
    """, unsafe_allow_html=True)

apply_theme(st.session_state.theme)

# -------------------- AGENT & API CONFIGURATION --------------------
@st.cache_data
def load_agents_config():
    """Load agents configuration from agents.yaml."""
    try:
        possible_paths = [
            Path(__file__).parent / "agents.yaml",
            Path("agents.yaml"),
            Path("/app/agents.yaml"),
            Path("./agents.yaml")
        ]
        
        for config_path in possible_paths:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
    except Exception as e:
        st.warning(f"Could not load agents.yaml: {e}. Using defaults.")

    return {
        'agents': {
            'Data Transformer': {
                'description': 'Transforms raw data into Markdown tables.',
                'default_prompt': 'Transform the following data into a clean Markdown table. Infer headers if missing.',
                'temperature': 0.1,
                'max_tokens': 4096
            },
            'Data Summarizer': {
                'description': 'Summarizes data and extracts keywords.',
                'default_prompt': 'Analyze the data and generate a summary. After the summary, write "Keywords:" followed by 5-7 important keywords.',
                'temperature': 0.4,
                'max_tokens': 2048
            },
            'Insight Extractor': {
                'description': 'Extracts key insights and patterns.',
                'default_prompt': 'From the data, identify significant insights, trends, or anomalies.',
                'temperature': 0.5,
                'max_tokens': 4096
            },
            'Follow-up Question Generator': {
                'description': 'Generates relevant follow-up questions.',
                'default_prompt': 'Based on the analysis, generate 3-5 insightful follow-up questions for further investigation.',
                'temperature': 0.6,
                'max_tokens': 2048
            }
        }
    }

# -------------------- HELPER FUNCTIONS --------------------
@st.cache_resource
def get_gemini_model():
    """Returns a cached instance of the Gemini model."""
    return genai.GenerativeModel("gemini-2.5-flash")

def execute_gemini_agent(prompt, data_context, temp, max_tok):
    """Executes a Gemini agent and returns (success, content)."""
    if not st.session_state.get('GEMINI_API_KEY'):
        return False, "Error: Gemini API key not configured."
    try:
        model = get_gemini_model()
        generation_config = genai.GenerationConfig(
            temperature=temp,
            max_output_tokens=max_tok
        )
        full_prompt = f"CONTEXT:\n{data_context}\n\n---\n\nTASK:\n{prompt}"
        response = model.generate_content(full_prompt, generation_config=generation_config)
        if not response.parts:
            return False, "API returned empty response. Check safety settings."
        return True, response.text
    except Exception as e:
        return False, f"API error: {str(e)}"

def parse_file_to_dataframe(file):
    """Parse uploaded file to pandas DataFrame."""
    try:
        if file.type == "text/csv":
            return pd.read_csv(file)
        elif file.type == "application/json":
            return pd.read_json(io.StringIO(file.getvalue().decode("utf-8")))
        else:
            content = file.getvalue().decode("utf-8")
            try:
                return pd.read_csv(io.StringIO(content))
            except:
                lines = content.split('\n')
                return pd.DataFrame({'content': lines})
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

def parse_text_to_dataframe(text):
    """Parse pasted text to pandas DataFrame."""
    try:
        try:
            return pd.read_csv(io.StringIO(text))
        except:
            pass
        
        try:
            return pd.read_json(io.StringIO(text))
        except:
            pass
        
        lines = text.split('\n')
        return pd.DataFrame({'content': [line for line in lines if line.strip()]})
    except Exception as e:
        st.error(f"Error parsing text: {e}")
        return None

def dataframe_to_json(df):
    """Convert DataFrame to JSON string."""
    return df.to_json(orient='records', indent=2)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("🎨 Select Theme")
    theme_cols = st.columns(3)
    theme_names = list(THEMES.keys())
    
    for idx, theme_name in enumerate(theme_names):
        col = theme_cols[idx % 3]
        with col:
            if st.button(f"{THEMES[theme_name]['icon']} {theme_name}", key=f"theme_{theme_name}"):
                st.session_state.theme = theme_name
                st.rerun()
    
    st.markdown("---")
    
    st.subheader("🔑 API Key")
    api_key_input = st.text_input(
        "Gemini API Key:",
        type="password",
        value=st.session_state.get('GEMINI_API_KEY', ''),
        help="Enter your Google Gemini API key"
    )
    if st.button("Set API Key"):
        st.session_state['GEMINI_API_KEY'] = api_key_input
        try:
            genai.configure(api_key=api_key_input)
            st.success("✅ API Key configured!")
        except Exception as e:
            st.error(f"❌ Configuration failed: {e}")
    
    st.markdown("---")
    
    st.subheader("📊 Progress")
    steps = {
        'config': '1️⃣ Configure',
        'upload': '2️⃣ Upload Data',
        'preview': '3️⃣ Preview',
        'json_edit': '4️⃣ Edit JSON',
        'analysis': '5️⃣ Analyze'
    }
    current_step_name = steps.get(st.session_state.current_step, 'Unknown')
    st.info(f"**Current Step:**\n\n{current_step_name}")

# Configure API if key exists
if st.session_state.get('GEMINI_API_KEY'):
    try:
        genai.configure(api_key=st.session_state['GEMINI_API_KEY'])
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
else:
    st.warning("⚠️ Please provide your Gemini API Key in the sidebar.")
    st.info("Get your API key at: https://makersuite.google.com/app/apikey")
    st.stop()

# -------------------- MAIN UI --------------------
theme = THEMES[st.session_state.theme]
st.title(f"{theme['icon']} Multi-Agent Analysis Hub")
st.markdown(f"<p style='text-align: center; color: {theme['secondary']}; font-size: 18px;'>Current Theme: <strong>{st.session_state.theme}</strong></p>", unsafe_allow_html=True)

agents_config = load_agents_config()
agents = agents_config.get('agents', {})

# -------------------- STEP 1: CONFIGURE NUMBER OF DATASETS --------------------
if st.session_state.current_step == 'config':
    st.markdown("<div class='step-indicator'>📋 Step 1: How many datasets do you want to analyze?</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        num_datasets = st.number_input(
            "Number of datasets:",
            min_value=1,
            max_value=10,
            value=st.session_state.num_datasets,
            step=1,
            help="Choose how many datasets you want to upload and analyze"
        )
        
        if st.button("✅ Confirm and Continue", type="primary", use_container_width=True):
            st.session_state.num_datasets = num_datasets
            st.session_state.datasets = [None] * num_datasets
            st.session_state.current_step = 'upload'
            st.rerun()

# -------------------- STEP 2: UPLOAD DATASETS --------------------
elif st.session_state.current_step == 'upload':
    st.markdown("<div class='step-indicator'>📤 Step 2: Upload or Paste Your Datasets</div>", unsafe_allow_html=True)
    
    st.info(f"📊 You will upload **{st.session_state.num_datasets}** dataset(s)")
    
    all_uploaded = True
    for i in range(st.session_state.num_datasets):
        with st.expander(f"📁 Dataset {i+1}", expanded=(st.session_state.datasets[i] is None)):
            st.markdown(f"<div class='dataset-card'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Upload File")
                uploaded_file = st.file_uploader(
                    f"Upload Dataset {i+1}",
                    type=["csv", "json", "txt"],
                    key=f"upload_{i}",
                    label_visibility="collapsed"
                )
                
                if uploaded_file:
                    df = parse_file_to_dataframe(uploaded_file)
                    if df is not None:
                        st.session_state.datasets[i] = {
                            'name': uploaded_file.name,
                            'data': df,
                            'source': 'file'
                        }
                        st.success(f"✅ Loaded: {uploaded_file.name}")
            
            with col2:
                st.subheader("Or Paste Data")
                pasted_text = st.text_area(
                    f"Paste data for Dataset {i+1}",
                    height=150,
                    key=f"paste_{i}",
                    placeholder="Paste CSV, JSON, or text data here...",
                    label_visibility="collapsed"
                )
                
                if st.button(f"Process Pasted Data", key=f"process_{i}"):
                    if pasted_text.strip():
                        df = parse_text_to_dataframe(pasted_text)
                        if df is not None:
                            st.session_state.datasets[i] = {
                                'name': f'Dataset_{i+1}_pasted',
                                'data': df,
                                'source': 'paste'
                            }
                            st.success(f"✅ Processed pasted data")
                            st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.session_state.datasets[i] is None:
                all_uploaded = False
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Configuration"):
            st.session_state.current_step = 'config'
            st.rerun()
    with col2:
        if all_uploaded:
            if st.button("➡️ Continue to Preview", type="primary"):
                st.session_state.current_step = 'preview'
                st.rerun()
        else:
            st.warning("⚠️ Please upload or paste all datasets to continue")

# -------------------- STEP 3: PREVIEW DATASETS --------------------
elif st.session_state.current_step == 'preview':
    st.markdown("<div class='step-indicator'>👁️ Step 3: Preview Your Datasets (First 10 Records)</div>", unsafe_allow_html=True)
    
    for i, dataset in enumerate(st.session_state.datasets):
        if dataset:
            st.subheader(f"📊 Dataset {i+1}: {dataset['name']}")
            
            df = dataset['data']
            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            
            st.dataframe(df.head(10), use_container_width=True)
            
            with st.expander(f"📈 Statistics for Dataset {i+1}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Column Types:**")
                    st.write(df.dtypes)
                with col2:
                    st.write("**Missing Values:**")
                    st.write(df.isnull().sum())
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Upload"):
            st.session_state.current_step = 'upload'
            st.rerun()
    with col2:
        if st.button("🔄 Transform to JSON", type="primary"):
            with st.spinner("Converting datasets to JSON..."):
                st.session_state.datasets_json = []
                for dataset in st.session_state.datasets:
                    if dataset:
                        json_str = dataframe_to_json(dataset['data'])
                        st.session_state.datasets_json.append({
                            'name': dataset['name'],
                            'json': json_str,
                            'edited_json': json_str
                        })
                st.session_state.current_step = 'json_edit'
                st.rerun()

# -------------------- STEP 4: EDIT JSON --------------------
elif st.session_state.current_step == 'json_edit':
    st.markdown("<div class='step-indicator'>✏️ Step 4: Review and Edit JSON Datasets</div>", unsafe_allow_html=True)
    
    for i, dataset_json in enumerate(st.session_state.datasets_json):
        st.subheader(f"📝 Dataset {i+1}: {dataset_json['name']}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Original JSON:**")
            st.json(dataset_json['json'])
        
        with col2:
            st.markdown("**Edit JSON:**")
            edited_json = st.text_area(
                f"Edit JSON for Dataset {i+1}",
                value=dataset_json['edited_json'],
                height=400,
                key=f"json_edit_{i}",
                label_visibility="collapsed"
            )
            st.session_state.datasets_json[i]['edited_json'] = edited_json
            
            try:
                json.loads(edited_json)
                st.success("✅ Valid JSON")
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON: {e}")
        
        st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Preview"):
            st.session_state.current_step = 'preview'
            st.rerun()
    with col2:
        all_valid = True
        for dataset_json in st.session_state.datasets_json:
            try:
                json.loads(dataset_json['edited_json'])
            except:
                all_valid = False
                break
        
        if all_valid:
            if st.button("🚀 Start Analysis", type="primary"):
                st.session_state.current_step = 'analysis'
                st.rerun()
        else:
            st.warning("⚠️ Please fix all JSON errors before continuing")

# -------------------- STEP 5: ANALYSIS --------------------
elif st.session_state.current_step == 'analysis':
    st.markdown("<div class='step-indicator'>🔬 Step 5: Multi-Agent Analysis</div>", unsafe_allow_html=True)
    
    if 'combined_data' not in st.session_state.workflow_data:
        combined_text = ""
        for i, dataset_json in enumerate(st.session_state.datasets_json):
            combined_text += f"\n\n=== DATASET {i+1}: {dataset_json['name']} ===\n"
            combined_text += dataset_json['edited_json']
        st.session_state.workflow_data['combined_data'] = combined_text
    
    with st.expander("📦 Combined Dataset Summary", expanded=False):
        st.text_area(
            "All datasets combined:",
            value=st.session_state.workflow_data['combined_data'],
            height=300,
            disabled=True
        )
    
    if 'summary' not in st.session_state.workflow_data:
        if st.button("📊 Generate Initial Summary", type="primary"):
            with st.spinner("Generating summary..."):
                summarizer = agents.get('Data Summarizer', {})
                success, summary_text = execute_gemini_agent(
                    summarizer.get('default_prompt', ''),
                    st.session_state.workflow_data['combined_data'],
                    summarizer.get('temperature', 0.4),
                    summarizer.get('max_tokens', 2048)
                )
                
                if success:
                    summary = summary_text
                    keywords = []
                    if "keywords:" in summary_text.lower():
                        parts = summary_text.split("Keywords:")
                        if len(parts) > 1:
                            summary = parts[0].strip()
                            keywords_str = parts[1].strip()
                            keywords = [k.strip() for k in keywords_str.split(',')]
                    
                    summary_html = summary
                    for keyword in keywords:
                        summary_html = summary_html.replace(keyword, f'<span class="keyword">{keyword}</span>')
                    
                    st.session_state.workflow_data['summary'] = summary_html
                    st.session_state.workflow_data['keywords'] = keywords
                    st.rerun()
                else:
                    st.error(f"❌ Summary generation failed: {summary_text}")
    
    if 'summary' in st.session_state.workflow_data:
        st.subheader("📊 Data Summary")
        st.markdown(
            f"<div class='results-box'>{st.session_state.workflow_data['summary']}</div>",
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        st.header("🤖 Agent Workflow Configuration")
        
        workflow_type = st.radio(
            "Select Workflow Type:",
            ["Single Agent", "Multi-Agent Sequence"],
            horizontal=True
        )
        
        agent_names = [name for name in agents.keys() 
                       if name not in ['Data Transformer', 'Data Summarizer']]
        
        if workflow_type == "Single Agent":
            if not agent_names:
                st.warning("⚠️ No agents available for single agent execution.")
            else:
                if 'selected_single_agent' not in st.session_state:
                    st.session_state.selected_single_agent = agent_names[0]
                
                if st.session_state.selected_single_agent not in agent_names:
                    st.session_state.selected_single_agent = agent_names[0]
                
                try:
                    current_index = agent_names.index(st.session_state.selected_single_agent)
                except ValueError:
                    current_index = 0
                    st.session_state.selected_single_agent = agent_names[0]
                
                selected_agent = st.selectbox(
                    "Select Agent:", 
                    agent_names,
                    index=current_index,
                    key="agent_selector"
                )
                
                if selected_agent != st.session_state.selected_single_agent:
                    st.session_state.selected_single_agent = selected_agent
                    st.rerun()
                
                agent_config = agents.get(selected_agent, {})
                
                with st.expander(f"⚙️ Configure '{selected_agent}'", expanded=True):
                    prompt = st.text_area(
                        "Prompt:",
                        value=agent_config.get('default_prompt', ''),
                        height=100,
                        key=f"single_prompt_{selected_agent}",
                        help="Customize the prompt for this agent"
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        temp = st.slider(
                            "Temperature:",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(agent_config.get('temperature', 0.5)),
                            step=0.05,
                            key=f"single_temp_{selected_agent}",
                            help="Controls randomness: 0 is focused, 1 is creative"
                        )
                    
                    with col2:
                        max_tok = st.number_input(
                            "Max Tokens:",
                            min_value=512,
                            max_value=8192,
                            value=int(agent_config.get('max_tokens', 4096)),
                            step=256,
                            key=f"single_tokens_{selected_agent}",
###

                            help="Maximum length of the response"
                        )
        
        # Execute button (outside expander for better visibility)
        st.markdown("---")
        
        if st.button(
            f"🚀 Execute {selected_agent}", 
            type="primary", 
            use_container_width=True,
            key=f"execute_single_{selected_agent}"
        ):
            # Validate we have data to process
            if 'combined_data' not in st.session_state.workflow_data:
                st.error("❌ No data available. Please generate a summary first.")
            else:
                with st.spinner(f"Running {selected_agent}..."):
                    success, result = execute_gemini_agent(
                        prompt,
                        st.session_state.workflow_data['combined_data'],
                        temp,
                        max_tok
                    )
                    
                    if success:
                        st.session_state.workflow_data['final_result'] = result
                        st.success("✅ Agent execution completed!")
                        st.rerun()
                    else:
                        st.error(f"❌ Execution failed: {result}")

        else:  # Multi-Agent Sequence
           # Multi-agent implementation remains the same
           selected_agents = st.multiselect(
           "Select agents in order:", 
           agent_names,
            help="Choose multiple agents to run in sequence"
         )
    
    if 'multi_agent_configs' not in st.session_state:
        st.session_state.multi_agent_configs = {}
    
    # Configure each selected agent
    for agent_name in selected_agents:
        agent_config = agents.get(agent_name, {})
        
        with st.expander(f"⚙️ Configure: {agent_name}", expanded=False):
            prompt = st.text_area(
                "Prompt:",
                value=agent_config.get('default_prompt', ''),
                height=100,
                key=f"multi_prompt_{agent_name}"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                temp = st.slider(
                    "Temperature:",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(agent_config.get('temperature', 0.5)),
                    step=0.05,
                    key=f"multi_temp_{agent_name}"
                )
            
            with col2:
                max_tok = st.number_input(
                    "Max Tokens:",
                    min_value=512,
                    max_value=8192,
                    value=int(agent_config.get('max_tokens', 4096)),
                    step=256,
                    key=f"multi_tokens_{agent_name}"
                )
            
            # Store configuration
            st.session_state.multi_agent_configs[agent_name] = {
                'prompt': prompt,
                'temp': temp,
                'max_tok': max_tok
            }
    
    # Execute multi-agent workflow
    st.markdown("---")
    
    if st.button(
        "🚀 Execute Multi-Agent Workflow", 
        type="primary", 
        use_container_width=True,
        disabled=len(selected_agents) == 0
    ):
        if not selected_agents:
            st.warning("⚠️ Please select at least one agent.")
        elif 'combined_data' not in st.session_state.workflow_data:
            st.error("❌ No data available. Please generate a summary first.")
        else:
            # Clear previous results
            st.session_state.workflow_data.pop('final_result', None)
            st.session_state.pop('multi_agent_steps', None)
            
            with st.spinner("Running multi-agent workflow..."):
                current_data = st.session_state.workflow_data['combined_data']
                st.session_state.multi_agent_steps = []
                workflow_failed = False
                
                # Execute agents in sequence
                for i, agent_name in enumerate(selected_agents):
                    st.info(f"▶️ Step {i+1}/{len(selected_agents)}: {agent_name}")
                    
                    config = st.session_state.multi_agent_configs.get(agent_name, {})
                    
                    success, result = execute_gemini_agent(
                        config.get('prompt', ''),
                        current_data,
                        config.get('temp', 0.5),
                        config.get('max_tok', 4096)
                    )
                    
                    if success:
                        st.session_state.multi_agent_steps.append({
                            'agent': agent_name,
                            'output': result
                        })
                        current_data = result  # Chain output to next agent
                        st.success(f"✅ Step {i+1} completed")
                    else:
                        st.error(f"❌ Failed at Step {i+1} ({agent_name}): {result}")
                        workflow_failed = True
                        break
                
                # Finalize workflow
                if not workflow_failed:
                    st.session_state.workflow_data['final_result'] = current_data
                    st.success("✅ Multi-agent workflow completed successfully!")
                    st.balloons()
                    st.rerun()
###
# -------------------- FINAL RESULT --------------------
if 'final_result' in st.session_state.workflow_data:
    st.markdown("---")
    st.header("🏆 Final Analysis Result")
    st.markdown(
        f"<div class='results-box'>{st.session_state.workflow_data['final_result']}</div>",
        unsafe_allow_html=True
    )
    
    # Download results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="📥 Download as Text",
            data=st.session_state.workflow_data['final_result'],
            file_name="analysis_result.txt",
            mime="text/plain"
        )
    with col2:
        # Create JSON export
        export_data = {
            'datasets': [ds['name'] for ds in st.session_state.datasets_json],
            'summary': st.session_state.workflow_data.get('summary', ''),
            'keywords': st.session_state.workflow_data.get('keywords', []),
            'final_result': st.session_state.workflow_data['final_result']
        }
        st.download_button(
            label="📥 Download as JSON",
            data=json.dumps(export_data, indent=2),
            file_name="analysis_result.json",
            mime="application/json"
        )
    
    # Follow-up questions section
    st.markdown("---")
    if st.button("❓ Generate Follow-up Questions", type="primary"):
        with st.spinner("Generating insightful follow-up questions..."):
            question_agent = agents.get('Follow-up Question Generator', {})
            success, questions = execute_gemini_agent(
                question_agent.get('default_prompt', ''),
                st.session_state.workflow_data['final_result'],
                question_agent.get('temperature', 0.6),
                question_agent.get('max_tokens', 2048)
            )
            if success:
                st.session_state.follow_up_questions = questions
                st.rerun()
            else:
                st.error(f"❌ Could not generate questions: {questions}")

# -------------------- FOLLOW-UP QUESTIONS --------------------
if 'follow_up_questions' in st.session_state:
    st.header("💡 Suggested Follow-up Questions")
    st.markdown(
        f"<div class='results-box'>{st.session_state.follow_up_questions}</div>",
        unsafe_allow_html=True
    )
    
    # Option to start new analysis with these questions
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Start New Analysis"):
            # Clear all data
            st.session_state.workflow_data = {}
            st.session_state.datasets = []
            st.session_state.datasets_json = []
            st.session_state.current_step = 'config'
            if 'multi_agent_steps' in st.session_state:
                del st.session_state.multi_agent_steps
            if 'follow_up_questions' in st.session_state:
                del st.session_state.follow_up_questions
            st.rerun()
    with col2:
        if st.button("📋 Copy Questions to Clipboard"):
            st.info("💡 Tip: Use Ctrl+C to copy the questions from above")

# Back button for analysis step
if st.session_state.current_step == 'analysis':
    st.markdown("---")
    if st.button("⬅️ Back to JSON Editor"):
        st.session_state.current_step = 'json_edit'
        st.rerun()

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: {theme['secondary']};'>"
    f"Built with ❤️ using <strong>Streamlit</strong> + <strong>Gemini API</strong> | "
    f"Multi-Agent System with Dynamic Themes</p>",
    unsafe_allow_html=True
)
