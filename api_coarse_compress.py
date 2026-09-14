"""Batch CLI for API-recommendation coarse compression datasets."""
import argparse
import json
from pathlib import Path

from api_coarse_compressor import APIRecommendationCoarseCompressor
from code_compressor import CodeCompressor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress new_first100-style API recommendation records; not standalone Python files.")
    parser.add_argument("--dataset", default="new_first100.json",
                        help="JSON array containing id, prompt, and gt fields.")
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--limit", type=int, help="Optional number of leading dataset records to process.")
    args = parser.parse_args()

    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Dataset must be a JSON array, such as new_first100.json.")
    if args.limit is not None:
        records = records[:args.limit]
    # Coarse AMI + ADF-IF does not enter the existing fine-grained entropy stage.
    scorer = CodeCompressor(args.model_name, device_map=args.device_map,
                            initialize_entropy_chunking=False)
    result = APIRecommendationCoarseCompressor(scorer).compress_dataset(records, args.token_budget)
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("dataset_records", "compressed_records", "skipped_records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
