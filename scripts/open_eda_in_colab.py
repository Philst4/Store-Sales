# scripts/open_eda_in_colab.py
import webbrowser

# --- Configuration ---
GITHUB_USERNAME = "philst4"
REPO_NAME = "store-sales"
BRANCH = "main"
NOTEBOOK_PATH = "eda/eda.ipynb"

def main():
    colab_url = f"https://colab.research.google.com/github/{GITHUB_USERNAME}/{REPO_NAME}/blob/{BRANCH}/{NOTEBOOK_PATH}"
    print("Opening in Colab (make sure you're signed into GitHub in your browser)...")
    webbrowser.open(colab_url)

if __name__ == "__main__":
    main()
