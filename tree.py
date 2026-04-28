from pathlib import Path

def print_tree(root: Path, prefix: str = ""):
    entries = sorted(
        root.iterdir(),
        key=lambda p: (p.is_file(), p.name.lower())
    )

    for i, path in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + path.name)

        if path.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(path, prefix + extension)


if __name__ == "__main__":
    print_tree(Path("configs"))