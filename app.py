import gradio as gr
import numpy as np
import os
import cv2
import uuid
from pathlib import Path

# Import the graph builder from your package
from cheque_processing_langgraph.__main__ import build_graph

# --- 1. Define the project root and load assets once on startup ---
try:
    project_root = str(Path(__file__).parent.resolve())
except NameError:
    project_root = os.getcwd()
print(f"Project root for Gradio app: {project_root}")

# Load the reference signature image from disk
ref_sig_path = Path(project_root) / "reference_signature.png"
if not ref_sig_path.exists():
    raise FileNotFoundError(f"FATAL: reference_signature.png not found at {ref_sig_path}")
REFERENCE_SIGNATURE_IMAGE = cv2.imread(str(ref_sig_path))

# --- 2. Build the LangGraph App, passing in the pre-loaded signature ---
print("Building the LangGraph application for the Gradio UI...")
app, text_llm = build_graph(reference_signature_image=REFERENCE_SIGNATURE_IMAGE)
print("LangGraph application built successfully.")


def process_cheque_with_ui(cheque_image_np: np.ndarray):
    """
    Main interface for the Gradio UI. It takes an image array, runs the graph,
    and returns the results. It no longer does any file I/O itself.
    """
    if cheque_image_np is None:
        return None, "## Error\n\nPlease upload a cheque image first."

    # Gradio provides an RGB numpy array; the graph needs BGR.
    cheque_image_bgr = cv2.cvtColor(cheque_image_np, cv2.COLOR_RGB2BGR)

    # --- 3. Create the CORRECT initial state for the graph (with image data) ---
    initial_state = {
        "image": cheque_image_bgr
    }

    print(f"Invoking graph with image data...")
    final_state = app.invoke(initial_state)
    
    # --- 4. Generate and return the report ---
    final_decision = final_state.get('final_decision', 'Error')
    feedback = "\n".join(final_state.get('feedback', ['An unknown error occurred.']))
    
    report = f"## Cheque Processing Report\n\n"
    report += f"**Final Decision:** `{final_decision}`\n\n"
    report += f"**Processing Feedback:**\n```\n{feedback}\n```\n\n"
    
    if final_state.get("audit_trail"):
        summary = final_state["audit_trail"].generate_llm_summary_report(text_llm)
        report += "### AI-Generated Audit Summary\n\n"
        report += summary
    else:
        report += "### Audit trail could not be generated due to a critical error."

    # Return the original image and the markdown report
    return cheque_image_np, report

# --- 5. Define the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Multi-Agent Cheque Processing System")
    gr.Markdown("Upload a cheque image to begin the automated extraction and fraud detection process.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Cheque")
            submit_button = gr.Button("Process Cheque", variant="primary")
        with gr.Column(scale=1):
            image_output = gr.Image(type="numpy", label="Cheque Preview")
            report_output = gr.Markdown(label="Processing Report")
            
    submit_button.click(
        fn=process_cheque_with_ui,
        inputs=[image_input],
        outputs=[image_output, report_output]
    )
    
    gr.Examples(
        examples=[os.path.join(project_root, "dbs_cheque.png")],
        inputs=[image_input],
    )

if __name__ == "__main__":
    demo.launch()