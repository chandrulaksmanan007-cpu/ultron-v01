import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from parser import extract_content
from rag_engine import RAGEngine
import time

st.set_page_config(page_title="ULTRON", layout="centered", page_icon="🤖")

# Clean, Modern UI CSS
st.markdown("""
<style>
/* Modern, Clean Theme */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Subtly style the sidebar */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(150, 150, 150, 0.1);
}

/* Popcorn Shredder Animation (Cleaned up) */
.shredder-container {
    text-align: center;
    height: 100px;
    position: relative;
    overflow: hidden;
    margin: 10px 0;
    background: rgba(150, 150, 150, 0.05);
    border-radius: 12px;
}
.popcorn-in {
    position: absolute;
    left: 45%;
    top: -30px;
    font-size: 24px;
    animation: dropIn 1s infinite linear;
}
.shredder-machine {
    position: absolute;
    left: 20%;
    top: 40px;
    font-size: 30px;
    width: 60%;
    border-top: 4px dashed rgba(150, 150, 150, 0.5);
    animation: chew 0.2s infinite alternate;
}
.popcorn-out {
    position: absolute;
    left: 45%;
    top: 70px;
    font-size: 16px;
    opacity: 0;
    animation: shootOut 1s infinite linear;
    animation-delay: 0.5s;
}

@keyframes dropIn { 0% { top: -30px; } 100% { top: 40px; opacity: 0; } }
@keyframes chew { 0% { transform: translateY(0px); } 100% { transform: translateY(2px); } }
@keyframes shootOut { 0% { top: 40px; opacity: 1; } 100% { top: 100px; opacity: 0; transform: scale(0.5); } }

/* Rocket Animation (Cleaned up) */
.rocket-wrapper { text-align: left; margin: 10px 0; height: 40px;}
.rocket { font-size: 30px; display: inline-block; }
.shaking { animation: shake 0.3s infinite; }
.launching { animation: launch 1s forwards ease-in; }

@keyframes shake {
    0% { transform: translate(1px, 1px) rotate(0deg); }
    20% { transform: translate(-1px, 0px) rotate(5deg); }
    40% { transform: translate(1px, -1px) rotate(-5deg); }
    60% { transform: translate(-1px, 1px) rotate(0deg); }
    80% { transform: translate(1px, -1px) rotate(5deg); }
    100% { transform: translate(1px, -1px) rotate(-5deg); }
}
@keyframes launch {
    0% { transform: translateY(0); opacity: 1; }
    100% { transform: translateY(-400px); opacity: 0; }
}

/* Welcome Screen */
.welcome-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 50vh;
    text-align: center;
}
.welcome-title {
    font-size: 4rem;
    font-weight: 900;
    margin-bottom: 10px;
    background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1.5px;
}
.welcome-subtitle {
    font-size: 1.2rem;
    color: #888;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# App Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

@st.cache_resource
def load_llm():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        torch_dtype=torch.float32
    )
    return tokenizer, model

with st.sidebar:
    st.title("ULTRON Data Core")
    
    with st.spinner("Loading Local Models..."):
        tokenizer, model = load_llm()
        
    if st.session_state.rag_engine is None:
        st.session_state.rag_engine = RAGEngine()
        
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    
    use_camera = st.checkbox("📸 Use Camera")
    camera_image = None
    if use_camera:
        camera_image = st.camera_input("Take a picture")
    
    all_files = list(uploaded_files) if uploaded_files else []
    if camera_image:
        all_files.append(camera_image)
        
    # Preview Section
    if all_files:
        with st.expander("👁️ Preview Uploaded Files"):
            for f in all_files:
                st.markdown(f"**{f.name}**")
                if f.type.startswith('image'):
                    st.image(f)
                elif f.name.endswith('.txt'):
                    text_content = f.getvalue().decode('utf-8', errors='ignore')
                    st.text_area("Content", text_content, height=150, disabled=True, key=f.name)
                elif f.name.endswith('.pdf'):
                    st.info("PDF Document indexed")
    
    if st.button("Index Documents", type="primary", disabled=not all_files, use_container_width=True):
        
        anim_placeholder = st.empty()
        anim_placeholder.markdown("""
        <div class="shredder-container">
            <div class="popcorn-in">🍿</div>
            <div class="shredder-machine">⚙️ ⬇️ ⚙️</div>
            <div class="popcorn-out">🍿🍿</div>
        </div>
        """, unsafe_allow_html=True)
        
        documents_to_index = []
        unreadable_files = []
        
        for file in all_files:
            if file.name in st.session_state.indexed_files:
                continue
                
            bytes_data = file.getvalue()
            text, is_readable = extract_content(bytes_data, file.name)
            
            if not is_readable:
                unreadable_files.append(file.name)
            else:
                documents_to_index.append({"filename": file.name, "content": text})
                st.session_state.indexed_files.append(file.name)
        
        if documents_to_index:
            success = st.session_state.rag_engine.index_documents(documents_to_index)
            if success:
                st.toast("Indexed successfully!", icon="✅")
                
        anim_placeholder.empty()
                
        if unreadable_files:
            st.error(f"Unreadable documents rejected: {', '.join(unreadable_files)}")
        elif documents_to_index:
            st.success("Indexing complete!")


# Main UI Logic
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">ULTRON</div>
        <div class="welcome-subtitle">Multimodal AI Assistant</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("ULTRON")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

prompt = st.chat_input("Ask ULTRON anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # We must use rerun here so that if it's the first message, the welcome screen clears cleanly
    # and the message appears in the standard chat log natively.
    st.rerun()

# If the last message was from the user, generate a response
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    
    prompt_text = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        context = ""
        if st.session_state.rag_engine:
            context = st.session_state.rag_engine.retrieve(prompt_text, k=2)
            
        messages = [
            {
                "role": "system", 
                "content": "You are ULTRON, a highly intelligent and accurate AI. Answer strictly based on the provided retrieved context. If the context is insufficient, state that clearly."
            },
            {
                "role": "user", 
                "content": f"Retrieved Context:\n{context}\n\nQuestion: {prompt_text}"
            }
        ]
        
        prompt_formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Rocket Shaking Animation
        rocket_placeholder = st.empty()
        rocket_placeholder.markdown("""
        <div class="rocket-wrapper">
            <div class="rocket shaking">🚀</div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            inputs = tokenizer([prompt_formatted], return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.1
            )
            
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            reply = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Rocket Launch Animation
            rocket_placeholder.markdown("""
            <div class="rocket-wrapper">
                <div class="rocket launching">🚀</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Briefly sleep so the user can see the rocket launch before text appears
            time.sleep(0.4)
            rocket_placeholder.empty()
            
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            
        except Exception as e:
            rocket_placeholder.empty()
            st.error(f"Error generating response: {str(e)}")
