import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

def initialize_model_and_tokenizer(model_id: str, use_quantization_config: bool = False, device: str = "cpu"):
    """
    Initializes and returns the tokenizer and language model.

    Parameters:
    - model_id (str): The model identifier to load.
    - use_quantization_config (bool): Whether to use 4-bit quantization for model loading.
    - device (str): The device to load the model on ('cpu' or 'gpu').

    Returns:
    - tokenizer: The tokenizer instance for the model.
    - llm_model: The language model instance.
    """
    # Check device validity
    if device not in ["cpu", "gpu"]:
        raise ValueError("Invalid device specified. Choose either 'cpu' or 'gpu'.")

    # Create quantization config for smaller model loading (optional)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if use_quantization_config else None

    # Setup Flash Attention 2 for faster inference if available
    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"
    print(f"[INFO] Using attention implementation: {attn_implementation}")

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

    # Instantiate the model
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.float16,  # datatype to use, we want float16
        quantization_config=quantization_config,
        low_cpu_mem_usage=False,  # use full memory
        attn_implementation=attn_implementation  # which attention version to use
    )

    # Move model to the specified device
    if device == "gpu" and not use_quantization_config:
        llm_model.to("cuda")
    elif device == "cpu":
        llm_model.to("cpu")

    return tokenizer, llm_model
