import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import InferenceModel


def main():
    parser = argparse.ArgumentParser(description="Run local inference with a trained model package.")
    parser.add_argument("--model", required=True, help="Path to trained model package ZIP.")
    parser.add_argument("--input", required=True, help="Path to input CSV or Excel file.")
    parser.add_argument("--output", default="outputs/predictions.csv", help="Output CSV path.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = InferenceModel(args.model)
    result_df = model.predict(args.input, include_input=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Prediction finished: {output_path}")
    print(result_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
