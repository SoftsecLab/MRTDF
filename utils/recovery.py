from pathlib import Path
from transformers import AutoModelForCausalLM
from config import Mistral_PATH, OUTPUT_MISTRAL_SAVE_DIR, require_path

def load_recovery_model(**kwargs):
    require_path(OUTPUT_MISTRAL_SAVE_DIR, 'OUTPUT_MISTRAL_SAVE_DIR')
    if (Path(OUTPUT_MISTRAL_SAVE_DIR) / 'adapter_config.json').is_file():
        require_path(Mistral_PATH, 'Mistral_PATH')
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(Mistral_PATH, **kwargs)
        return PeftModel.from_pretrained(base, OUTPUT_MISTRAL_SAVE_DIR)
    return AutoModelForCausalLM.from_pretrained(OUTPUT_MISTRAL_SAVE_DIR, **kwargs)
