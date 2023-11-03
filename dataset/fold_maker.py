import random

lines = []
with open("piid_name_train.txt") as f:
    lines.extend(f.readlines())

# with open("piid_name_test.txt") as f:
#     lines.extend(f.readlines())

with open("piid_name_val.txt") as f:
    lines.extend(f.readlines())
# lines = [i.strip() for i in lines]
random.shuffle(lines)
size = len(lines) // 5
for i in range(5):
    with open(f"folds/fold_{i}.txt", 'w') as f:
        start = int(i*len(lines)/5)
        end = int((i+1)*len(lines)/5)
        f.writelines(lines[start:end])
