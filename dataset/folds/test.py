from pathlib import Path
total = 0
for path in Path('.').iterdir():
    if path.suffix.endswith("txt"):
        with open(path, 'r') as f:
            tmp = len(f.readlines())
            total += tmp
            print(path.name, tmp)
print(total)