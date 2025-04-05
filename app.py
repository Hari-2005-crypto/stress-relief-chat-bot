from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import gradio as gr
import json
from datetime import datetime
import requests
from PIL import Image
import io
import random

# Global variable to store conversation history
conversation_history = []

# Download LPU logo
def get_lpu_logo():
    logo_url = "https://www.lpu.in/images/logo.png"
    response = requests.get(logo_url)
    return Image.open(io.BytesIO(response.content))

def save_conversation_history():
    with open("conversation_history.json", "w") as f:
        json.dump(conversation_history, f)

def load_conversation_history():
    global conversation_history
    try:
        with open("conversation_history.json", "r") as f:
            conversation_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        conversation_history = []

def initialize_llm():
    groq_api_key = os.getenv("GROQ_API_KEY", "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")
    return ChatGroq(
        temperature=0.7,
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192"
    )

def create_vector_db():
    pdf_path = r"C:\Users\Bhagavan\Downloads\stress_management.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    
    chroma_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    vector_db.persist()
    print("ChromaDB created and data saved")
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    prompt_template = """You are ZenBot, a stress management coach at LPU. Provide practical, science-backed techniques to help students manage stress in a friendly, supportive tone. Always include emojis in your responses to make them more engaging.
    
    Context:
    {context}
    
    Current conversation:
    Student: {question}
    ZenBot: """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=['context', 'question']
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

def update_history(user_input, bot_response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_history.append({
        "timestamp": timestamp,
        "user": user_input,
        "bot": bot_response
    })
    save_conversation_history()
    return conversation_history

# Stress relief tips for the sidebar
stress_tips = [
    "🌿 Try the 4-7-8 breathing technique: Inhale 4s, hold 7s, exhale 8s",
    "💧 Stay hydrated - dehydration increases cortisol (stress hormone)",
    "🔄 Take a 5-minute walk every hour to reset your mind",
    "🎧 Listen to binaural beats for 15 minutes to reduce anxiety",
    "✍️ Journal your thoughts for 5 minutes to declutter your mind",
    "🌅 Morning sunlight exposure regulates your circadian rhythm",
    "🍵 Sip chamomile tea - it contains apigenin, a natural relaxant"
]

def get_random_tip():
    return random.choice(stress_tips)

print("Initializing ZenBot - LPU Stress Relief Companion...")
try:
    # Load LPU logo
    lpu_logo = get_lpu_logo()
    lpu_logo_path = "lpu_logo.png"
    lpu_logo.save(lpu_logo_path)
    
    load_conversation_history()
    llm = initialize_llm()
    
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    if not os.path.exists(db_path):
        print("Creating new vector database...")
        vector_db = create_vector_db()
    else:
        print("Loading existing vector database...")
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
    
    qa_chain = setup_qa_chain(vector_db, llm)
    
    def chatbot_response(message, chat_history):
        if not message.strip():
            return "🧘 Please share what's stressing you today. I'm here to help!"
        
        try:
            result = qa_chain({"query": message})
            response = f"{result['result']}\n\n💡 Remember: {get_random_tip()}"
            update_history(message, response)
            return response
        except Exception as e:
            error_msg = f"🛠️ I'm having technical difficulties. Please try again later. ({str(e)})"
            update_history(message, error_msg)
            return error_msg

    # Enhanced Dark Theme CSS with animated elements
    custom_css = """
    :root {
        --primary: #8a2be2;
        --secondary: #9370db;
        --accent: #ba55d3;
        --dark: #121212;
        --darker: #0a0a0a;
        --light: #e0e0e0;
        --lighter: #f5f5f5;
    }
    
    .gradio-container {
        background: var(--darker) !important;
        color: var(--light) !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .chatbot {
        min-height: 650px;
        border-radius: 16px !important;
        background: var(--dark) !important;
        border: 1px solid #333 !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .chatbot .user {
        background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%) !important;
        color: white !important;
        border-radius: 18px 18px 0 18px !important;
        max-width: 85%;
        margin-left: auto;
        border: none !important;
        padding: 12px 16px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .chatbot .assistant {
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%) !important;
        color: white !important;
        border-radius: 18px 18px 18px 0 !important;
        max-width: 85%;
        border: none !important;
        padding: 12px 16px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .history-column {
        min-height: 650px !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 12px !important;
    }
    
    .history-panel {
        flex-grow: 1 !important;
        overflow-y: auto !important;
        max-height: 500px !important;
        background: var(--dark) !important;
        color: var(--light) !important;
        border-radius: 16px !important;
        padding: 16px !important;
        border: 1px solid #333 !important;
        box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.2);
    }
    
    .team-details {
        background: var(--dark) !important;
        color: var(--light) !important;
        padding: 16px;
        border-radius: 16px;
        border: 1px solid #333 !important;
        margin-top: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        padding: 20px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: none !important;
    }
    
    .footer {
        background: var(--dark) !important;
        color: var(--light) !important;
        padding: 16px;
        border-radius: 16px;
        margin-top: 20px;
        border: 1px solid #333 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .btn-primary {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .btn-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(138, 43, 226, 0.3) !important;
    }
    
    .btn-secondary {
        background: var(--dark) !important;
        color: var(--light) !important;
        border: 1px solid var(--secondary) !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .btn-secondary:hover {
        background: rgba(138, 43, 226, 0.1) !important;
        transform: translateY(-2px) !important;
    }
    
    .textbox {
        border-radius: 16px !important;
        padding: 16px !important;
        background: var(--dark) !important;
        color: var(--light) !important;
        border: 1px solid #333 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .textbox:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(138, 43, 226, 0.3) !important;
    }
    
    .example {
        background: rgba(138, 43, 226, 0.1) !important;
        color: var(--light) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        transition: all 0.3s ease !important;
    }
    
    .example:hover {
        background: rgba(138, 43, 226, 0.2) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(138, 43, 226, 0.2) !important;
    }
    
    .tip-card {
        background: linear-gradient(135deg, #182848 0%, #4b6cb7 100%);
        color: white;
        padding: 16px;
        border-radius: 16px;
        margin-bottom: 12px;
        border: none !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .logo {
        border-radius: 16px;
        border: 2px solid var(--primary);
        padding: 4px;
        background: white;
    }
    
    .divider {
        border-top: 1px solid #333;
        margin: 16px 0;
        opacity: 0.5;
    }
    
    .accordion {
        background: var(--dark) !important;
        color: var(--light) !important;
        border: 1px solid #333 !important;
        border-radius: 16px !important;
    }
    
    .accordion-item {
        background: var(--dark) !important;
    }
    
    .tab-button {
        background: var(--dark) !important;
        color: var(--light) !important;
    }
    """

    with gr.Blocks(theme=gr.themes.Default(
        primary_hue="purple",
        secondary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Poppins"), "Arial", "sans-serif"]
    ), title="ZenBot - LPU Stress Relief", css=custom_css) as app:
        
        # Header section with animated gradient
        with gr.Row(equal_height=True, variant="panel"):
            with gr.Column(scale=2):
                gr.Markdown("""
                <div class="header">
                <h1 style="margin: 0; font-weight: 700;">🧘‍♂️ ZenBot - LPU Stress Relief Companion</h1>
                <p style="margin: 0; opacity: 0.9; font-size: 1.1em;">Your AI-powered guide to academic calm and focus</p>
                </div>
                """)
            with gr.Column(scale=1, min_width=150):
                gr.Image(lpu_logo_path, 
                        label="Lovely Professional University", 
                        width=140, 
                        show_label=False, 
                        elem_classes="logo")
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("""
                <div class="team-details">
                <h3 style="margin-top: 0; color: var(--primary);">🌟 Development Team</h3>
                <p>👨‍💻 C.Hari Sri Charan (12322849)</p>
                <p>👨‍💻 M.Vittal (12307384)</p>
                <p>👨‍💻 S.Deepak (12308708)</p>
                </div>
                """)
        
        # Main content area
        with gr.Row():
            # Left sidebar with history and tips
            with gr.Column(scale=1, min_width=320, elem_classes="history-column"):
                with gr.Accordion("💡 Daily Stress Relief Tip", open=True):
                    gr.Markdown(f"""
                    <div class="tip-card">
                    <h3 style="margin-top: 0;">{get_random_tip()}</h3>
                    </div>
                    """)
                
                with gr.Accordion("📜 Conversation History", open=True):
                    history_display = gr.JSON(
                        value=conversation_history,
                        label="Past Conversations",
                        container=True,
                        elem_classes="history-panel"
                    )
                    
                    with gr.Row():
                        clear_btn = gr.Button("✨ Clear History", variant="secondary")
                        new_chat_btn = gr.Button("🆕 New Session", variant="primary")
                        
                def clear_history():
                    global conversation_history
                    conversation_history = []
                    save_conversation_history()
                    return []
                    
                clear_btn.click(
                    fn=clear_history,
                    outputs=history_display
                )
                
                def new_chat():
                    return []
                    
                new_chat_btn.click(
                    fn=new_chat,
                    outputs=[history_display]
                )
            
            # Main chat interface
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=650,
                    elem_classes="chatbot",
                    show_copy_button=True,
                    avatar_images=(
                        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
                        "https://cdn-icons-png.flaticon.com/512/4712/4712139.png"
                    ),
                    bubble_full_width=False
                )
                
                msg = gr.Textbox(
                    label="What's on your mind?", 
                    placeholder="Share your stress or ask for relief techniques...",
                    elem_classes="textbox",
                    container=False,
                    lines=3
                )
                
                with gr.Row():
                    clear = gr.ClearButton([msg, chatbot], 
                                         value="🧹 Clear Chat", 
                                         variant="secondary")
                    submit_btn = gr.Button("🚀 Send", 
                                         variant="primary")
                
                examples = gr.Examples(
                    examples=[
                        ["I'm overwhelmed with my exam schedule 😫"],
                        ["How can I relax quickly before a presentation?"],
                        ["I can't sleep because of academic pressure"],
                        ["What are some effective study breaks?"],
                        ["I feel anxious about my grades"]
                    ],
                    inputs=msg,
                    label="Common Stress Scenarios:",
                    examples_per_page=5
                )
                
                def respond(message, chat_history):
                    bot_message = chatbot_response(message, chat_history)
                    chat_history.append((message, bot_message))
                    update_history(message, bot_message)
                    return "", chat_history
                
                submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
                msg.submit(respond, [msg, chatbot], [msg, chatbot])
                
                # Footer with animated resources
                gr.Markdown("""
                <div class="footer">
                <h3 style="margin-top: 0; color: var(--primary);">🌱 Campus Wellness Resources</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                <div>
                <p><b>📍 LPU Wellness Center</b></p>
                <p>• Block 32, Room 205</p>
                <p>• Open 9AM-5PM Mon-Sat</p>
                </div>
                <div>
                <p><b>📞 Emergency Contacts</b></p>
                <p>• Campus Security: 01824-404040</p>
                <p>• 24/7 Counselor: 1800-123-456</p>
                </div>
                </div>
                <p style="font-size: 0.8em; opacity: 0.7; margin-bottom: 0;">Remember: Your mental health matters 💜</p>
                </div>
                """)
        
        # Update history panel when chat changes
        chatbot.change(
            fn=lambda: conversation_history,
            outputs=history_display
        )
    
    app.launch()

except Exception as e:
    print(f"Failed to initialize chatbot: {str(e)}")