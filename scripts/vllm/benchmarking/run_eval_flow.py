import argparse
import os
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(
        description="Run MMLU-Pro evaluation flow.")
    parser.add_argument("--limit",
                        type=int,
                        default=5,
                        help="Limit number of samples per category")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="Path to save the output JSON results")
    args = parser.parse_args()

    model_path = "/home/patemotter_google_com/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1"
    port = 8000

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "/home/patemotter_google_com/uv_venv/bin/lm_eval", "--model",
            "local-chat-completions", "--model_args",
            f"model={model_path},base_url=http://localhost:{port}/v1/chat/completions,num_concurrent=32",
            "--tasks", "mmlu_pro", "--limit",
            str(args.limit), "--apply_chat_template", "--gen_kwargs",
            '{"chat_template_kwargs": {"enable_thinking": false}}',
            "--output_path", tmpdir
        ]

        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Find the results JSON file in the tmpdir.
        results_file = None
        for f in os.listdir(tmpdir):
            if f.endswith(".json") and "results" in f:
                results_file = os.path.join(tmpdir, f)
                break

        if not results_file:
            raise FileNotFoundError(
                "Could not find results JSON file from lm_eval output.")

        print(f"Copying results from {results_file} to {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                    exist_ok=True)
        shutil.copy(results_file, args.output)


if __name__ == "__main__":
    main()
