from datasets import load_dataset

dataset = load_dataset('quora', split='train[240000:320000]')

print(dataset[:5])