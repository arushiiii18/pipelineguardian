import json, os, re

src = os.path.expanduser(
    r"~/OneDrive/Documents/research/leakage-analysis/benchmark_test/sample_notebooks"
)

ids = [
    "2021-09-03-nb_2630",
    "2021-09-05-nb_1162",
    "2021-09-05-nb_2770",
    "2021-09-01-nb_604",
    "2021-09-03-nb_891",
    "2021-09-04-nb_2749",
]

pat = re.compile(
    r"train_test_split|test_size|test_|_test|test\.|concat|merge|join|intersection|"
    r"duplic|sample\(|split|fold|KFold|Stratified|Group|validation",
    re.I,
)

for nb_id in ids:
    path = os.path.join(src, nb_id + ".ipynb")
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    print(f"\n===== {nb_id} =====")

    for cell_no, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        for line_no, line in enumerate(cell.get("source", []), 1):
            if pat.search(line):
                print(f"cell {cell_no}, line {line_no}: {line.strip()}")
