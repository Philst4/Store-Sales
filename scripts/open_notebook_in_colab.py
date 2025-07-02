import webbrowser
import argparse

# --- Configuration ---
GITHUB_USERNAME = "philst4"
REPO_NAME = "store-sales"
BRANCH = "main"
NOTEBOOK_PATH = "notebooks"

def main(args):
    notebook_name = f"{args.notebook}.ipynb"
    colab_url = f"https://colab.research.google.com/github/{GITHUB_USERNAME}/{REPO_NAME}/blob/{BRANCH}/{NOTEBOOK_PATH}/{notebook_name}"
    print(f"Opening '{notebook_name}' in Colab (make sure you're signed into GitHub in your browser)...")
    webbrowser.open(colab_url)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for opening (specified) notebook in Colab")
    parser.add_argument("--notebook", type=str, default="eda", help="Which notebook to open in Colab (refer to 'notebooks' directory)") 
    args = parser.parse_args()
    assert args.notebook in ("eda")
    main(args)
