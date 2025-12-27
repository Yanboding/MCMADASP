import matplotlib
import glob
import os
pattern = os.path.join("results", "determine_warmup", "*.pickle")
jsonl_files = glob.glob(pattern)
for file_path in jsonl_files:
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            self.load(data)
if __name__=="__main__":
    print('hello world')