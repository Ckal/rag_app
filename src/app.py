import os.path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gradio as gr
import torch


if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

model_name = "Qwen/Qwen2.5-7B-Instruct"

# # Configure 4-bit quantization
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,                      # Enable 4-bit loading
#     bnb_4bit_compute_dtype=torch.float16,   # Compute in float16
#     bnb_4bit_use_double_quant=True,        # Double quantization for even smaller size
#     bnb_4bit_quant_type="nf4"              # Normal Float 4-bit
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    # quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)


def generate_response(user_input):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize and move to device
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )

    # Extract only the new tokens
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode to text
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Your Question", placeholder="Ask me anything..."),
    outputs=gr.Textbox(label="Response"),
    title="Qwen 2.5-7B Chatbot",
    description="Ask questions and get responses from Qwen 2.5-7B model"
)

demo.launch()



