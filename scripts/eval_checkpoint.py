import argparse
import json
from pathlib import Path

from cw360.inference.load import load_inference_model
from cw360.inference.generate import generate_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a prompt set.")
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint.")
    parser.add_argument("--random-model", action="store_true", help="Use random weights instead of a checkpoint.")
    parser.add_argument("--prompt-file", type=str, required=True, help="JSONL file with prompts.")
    parser.add_argument("--output-dir", type=str, default="eval_outputs", help="Directory to save eval results.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    if not args.checkpoint and not args.random_model:
        parser.error("Either --checkpoint or --random-model must be provided.")
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, tokenizer = load_inference_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        random_weights=args.random_model,
    )
    
    metadata = {}
    if args.checkpoint:
        try:
            from cw360.checkpoint.metadata import metadata_path_for_checkpoint, read_metadata_json
            ckpt_path = Path(args.checkpoint)
            meta_path = metadata_path_for_checkpoint(ckpt_path)
            if meta_path.exists():
                md = read_metadata_json(meta_path)
                metadata = md.to_dict()
        except ImportError:
            pass
            
    prompts_data = []
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts_data.append(json.loads(line))
                
    results = []
    for p_data in prompts_data:
        prompt_text = p_data["prompt"]
        output_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
        )
        res = {
            "prompt_id": p_data.get("prompt_id", "unknown"),
            "prompt": prompt_text,
            "generated": output_text,
        }
        results.append(res)
        
    out_name = Path(args.checkpoint).stem if args.checkpoint else "random_model"
    jsonl_path = output_dir / f"{out_name}_eval.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    md_path = output_dir / f"{out_name}_eval.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Evaluation Report: {out_name}\n\n")
        f.write("## Metadata\n")
        if metadata:
            f.write(f"- Model Size Label: {metadata.get('model_size_label', 'N/A')}\n")
            f.write(f"- Data Snapshot: {metadata.get('data_snapshot', 'N/A')}\n")
            tpu_cursor = metadata.get("tpu_data_cursor") or {}
            f.write(f"- Prepacked Dataset ID: {tpu_cursor.get('prepacked_dataset_id', 'N/A')}\n")
        else:
            f.write("No metadata found (or random model).\n")
        f.write("\n## Generations\n")
        for r in results:
            f.write(f"### Prompt: {r['prompt_id']}\n")
            f.write(f"**Prompt:** `{r['prompt']}`\n\n")
            f.write("**Generated:**\n```python\n")
            f.write(r['generated'] + "\n```\n\n")
            
    print(f"Eval finished. Saved to {jsonl_path} and {md_path}")

if __name__ == "__main__":
    main()
