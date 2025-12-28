import chz
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.format_colorized import format_colorized
from k8s.k8s_data import load_gemini_history

@chz.chz
class Config:
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str = "qwen3"
    dataset_path: str = "k8s-dataset.jsonl"
    max_length: int = 4096
    generate_sub_traj: bool = False
    batch_size: int = 4
    strip_thinking_from_history: bool = False

def run(cfg: Config):
    print(f"Loading dataset from {cfg.dataset_path}...")
    tokenizer = get_tokenizer(cfg.model_name)
    renderer = renderers.get_renderer(cfg.renderer_name, tokenizer)
    # See https://tinker-docs.thinkingmachines.ai/rl/sequence-extension
    # for more details on why this is needed.
    renderer.strip_thinking_from_history = cfg.strip_thinking_from_history
    
    # Load dataset using our custom loader
    ds_wrapper = load_gemini_history(
        tokenizer, 
        renderer, 
        cfg.dataset_path, 
        generate_sub_traj=cfg.generate_sub_traj,
        batch_size=cfg.batch_size
    )
    dataset = ds_wrapper.hf_dataset
    print(f"Loaded {len(dataset)} examples.")
    
    # Iterate over batches
    for i in range(len(dataset)):
        try:
            batch_datums = ds_wrapper.get_batch(i)
        except IndexError:
            break
            
        if not batch_datums:
            break
            
        for datum in batch_datums:
            int_tokens = list(datum.model_input.to_ints())
            weights = datum.loss_fn_inputs["weights"].tolist()
            
            # Align weights with tokens if needed
            if len(weights) < len(int_tokens):
                weights = [0.0] * (len(int_tokens) - len(weights)) + weights
            elif len(weights) > len(int_tokens):
                weights = weights[:len(int_tokens)]
                
            print("-" * 80)
            print(format_colorized(int_tokens, weights, tokenizer))
            print("-" * 80)
            
            user_input = input("Press Enter for next example, 'q' to quit: ")
            if user_input.lower() == 'q':
                return

if __name__ == "__main__":
    chz.nested_entrypoint(run, allow_hyphens=True)
