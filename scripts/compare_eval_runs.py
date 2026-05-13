import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare eval runs.")
    parser.add_argument("eval_files", nargs="+", help="Paths to eval JSONL files.")
    parser.add_argument("--output", type=str, default="eval_outputs/comparison.md", help="Output markdown path.")
    
    args = parser.parse_args()
    
    runs = {}
    prompt_ids = []
    
    for ef in args.eval_files:
        ef_path = Path(ef)
        run_name = ef_path.stem
        runs[run_name] = {}
        with ef_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    p_id = data.get("prompt_id", "unknown")
                    runs[run_name][p_id] = data
                    if p_id not in prompt_ids:
                        prompt_ids.append(p_id)
                        
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    run_names = list(runs.keys())
    
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Eval Comparison\n\n")
        
        f.write("| Prompt ID | " + " | ".join(run_names) + " |\n")
        f.write("|" + "|".join(["---"] * (len(run_names) + 1)) + "|\n")
        
        for p_id in prompt_ids:
            row = [p_id]
            for r_name in run_names:
                data = runs[r_name].get(p_id)
                if data:
                    gen = data.get("generated", "")
                    gen = gen.replace("\n", "<br>").replace("|", "\\|")
                    row.append(gen)
                else:
                    row.append("N/A")
            f.write("| " + " | ".join(row) + " |\n")
            
    print(f"Comparison saved to {out_path}")

if __name__ == "__main__":
    main()
