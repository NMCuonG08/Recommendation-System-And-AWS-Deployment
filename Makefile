install:
	uv sync --all-groups
	uv run python -m ipykernel install --user --name=recommend-system --display-name="Python (recommend-system)"

run-notebook:
	uv run jupyter lab --no-browser

# Download Kaggle dataset (hqinsiders/hq-trivia-sample) into kagglehub cache.
# Requires KAGGLE_USERNAME / KAGGLE_KEY env vars (or ~/.kaggle/kaggle.json).
kaggle-download:
	uv run python -c "import kagglehub; print(kagglehub.dataset_download('hqinsiders/hq-trivia-sample'))"