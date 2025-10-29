#!/usr/bin/env python3
"""
Chat-like UI for Document Analyzer using Gradio
"""
import gradio as gr
import sys
import os
import json
import tempfile
from typing import List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from orchestrator import DocumentOrchestrator

class DocumentChatBot:
    def __init__(self):
        self.current_document = None
        self.document_name = None
        self.orchestrator = None

    def set_api_key(self, api_key: str) -> str:
        """Set OpenAI API key"""
        if api_key.strip():
            self.orchestrator = DocumentOrchestrator(api_key.strip())
            return "✅ API key set successfully!"
        return "❌ Please enter a valid API key"

    def upload_document(self, file) -> str:
        """Handle document upload"""
        if file is None:
            return "❌ No file uploaded"

        try:
            self.current_document = file.name
            self.document_name = os.path.basename(file.name)
            return f"✅ Document '{self.document_name}' uploaded successfully! You can now ask questions about it."
        except Exception as e:
            return f"❌ Upload failed: {str(e)}"

    def chat(self, message: str, history: List[dict]) -> Tuple[List[dict], str]:
        """Process chat message"""
        if not self.orchestrator:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "❌ Please set your OpenAI API key first."})
            return history, ""

        if not self.current_document:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "❌ Please upload a document first."})
            return history, ""

        if not message.strip():
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "❌ Please enter a question."})
            return history, ""

        try:
            # Add user message
            history.append({"role": "user", "content": message})

            # Run analysis
            result = self.orchestrator.analyze(message, self.current_document, "temp_out")

            # Format response
            response = self.format_response(result)
            history.append({"role": "assistant", "content": response})

        except Exception as e:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"❌ Analysis failed: {str(e)}"})

        return history, ""

    def format_response(self, result: dict) -> str:
        """Format analysis result for chat display"""
        if "error" in result:
            return f"❌ Error: {result['error']}"

        # Parse result
        answer_text = result.get("answer", "No answer generated")
        confidence = result.get("confidence", 0)
        summary = result.get("summary", "")
        recommendations = result.get("recommendations", [])

        # Handle JSON responses
        if isinstance(answer_text, str) and "```json" in answer_text:
            try:
                json_start = answer_text.find("```json") + 7
                json_end = answer_text.find("```", json_start)
                json_str = answer_text[json_start:json_end].strip()
                answer_data = json.loads(json_str)

                answer_text = answer_data.get("answer", answer_text)
                confidence = answer_data.get("confidence", confidence)
                summary = answer_data.get("summary", summary)
                recommendations = answer_data.get("recommendations", recommendations)
            except:
                pass

        # Build formatted response
        response = f"🎯 **Answer:** {answer_text}\n\n"

        # Add confidence with emoji
        if confidence >= 0.8:
            conf_emoji = "🟢"
        elif confidence >= 0.5:
            conf_emoji = "🟡"
        else:
            conf_emoji = "🔴"

        response += f"{conf_emoji} **Confidence:** {confidence:.1%}\n\n"

        if summary:
            response += f"📋 **Summary:** {summary}\n\n"

        if recommendations:
            response += "💡 **Recommendations:**\n"
            for i, rec in enumerate(recommendations, 1):
                response += f"{i}. {rec}\n"

        return response

    def clear_chat(self) -> Tuple[List[dict], str]:
        """Clear chat history"""
        return [], ""

def create_interface():
    """Create Gradio interface"""
    bot = DocumentChatBot()

    with gr.Blocks(title="Document Analyzer Chat", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 📄 Document Analyzer Chat
        Upload a document and chat with it! Ask questions in natural language about the content.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Setup")

                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    info="Enter your OpenAI API key"
                )

                api_key_btn = gr.Button("Set API Key", variant="secondary")
                api_key_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### 📤 Upload Document")

                file_upload = gr.File(
                    label="Upload Document Image",
                    file_types=[".png", ".jpg", ".jpeg", ".bmp", ".tiff"],
                    type="filepath"
                )

                upload_status = gr.Textbox(label="Upload Status", interactive=False)

                gr.Markdown("### 💡 Example Questions")
                gr.Markdown("""
                - What type of document is this?
                - Extract all names and dates
                - What are the main points?
                - Find all monetary amounts
                - Extract table data
                """)

            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat with Your Document")

                chatbot = gr.Chatbot(
                    height=500,
                    label="Document Analysis Chat",
                    show_label=False,
                    type="messages"
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about your document...",
                        label="Message",
                        show_label=False,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        # Event handlers
        api_key_btn.click(
            bot.set_api_key,
            inputs=[api_key_input],
            outputs=[api_key_status]
        )

        file_upload.change(
            bot.upload_document,
            inputs=[file_upload],
            outputs=[upload_status]
        )

        def respond_and_clear(message, history):
            new_history, _ = bot.chat(message, history)
            return new_history, ""

        send_btn.click(
            respond_and_clear,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )

        msg_input.submit(
            respond_and_clear,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )

        clear_btn.click(
            bot.clear_chat,
            outputs=[chatbot, msg_input]
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )