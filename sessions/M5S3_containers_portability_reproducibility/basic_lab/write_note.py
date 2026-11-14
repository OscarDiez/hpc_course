from pathlib import Path

# /work is the host folder selected by the bind option.
print("Container folder:", Path.cwd())
print("Training program visible:", Path("train_model.py").is_file())

# The host will keep this file after the container process ends.
Path("from_container.txt").write_text("Hello from inside the container.\n")
