from datasets import load_dataset


def add_prompt(example):
    example["prompt"] = f"Question: {example['question']} Answer:"
    return example


dataset = load_dataset('wiki_qa')

for split in dataset.keys():
    dataset[split] = dataset[split].map(add_prompt)
    dataset[split].save_to_disk(f'datasets/wikiqa_{split}.hf')
