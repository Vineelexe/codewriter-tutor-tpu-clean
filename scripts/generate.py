import argparse
from pathlib import Path

from cw360.inference.load import load_inference_model
from cw360.inference.generate import generate_text

def main() -> None:
    parser = argparse.ArgumentParser(description="CodeWriter-Tutor generation CLI.")
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint.")
    parser.add_argument("--random-model", action="store_true", help="Use random weights instead of a checkpoint.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt string.")
    parser.add_argument("--prompt-file", type=str, default=None, help="Path to file containing prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-cache", action="store_true", help="Disable KV cache.")
    
    args = parser.parse_args()
    
    if args.prompt is None and args.prompt_file is None:
        parser.error("Either --prompt or --prompt-file must be provided.")
        
    if args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        
    if not args.checkpoint and not args.random_model:
        parser.error("Either --checkpoint or --random-model must be provided.")
        
    model, tokenizer = load_inference_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        random_weights=args.random_model,
    )
    
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        use_cache=not args.no_cache,
        device=args.device,
    )
    
    print(output)

if __name__ == "__main__":
    main()
