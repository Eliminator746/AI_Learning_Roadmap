import os
docs_dir="E:\Retrieval_RAG\data"
docs_dir = os.path.abspath(docs_dir)
txt_files = [f for f in os.listdir(docs_dir) if f.endswith(".txt")]

for file in sorted(txt_files):
    filename = os.path.join(docs_dir,file)
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    print(content)
    break
