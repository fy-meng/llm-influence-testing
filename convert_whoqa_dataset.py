import json
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the JSON file
with open("WhoQA/WhoQA.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Prepare a list for Hugging Face dataset format
hf_data = []

print('len', len(data))
data = data[:len(data) // 100]
print('new len', len(data))

for entry in tqdm(data):
    question_list = entry["questions"]
    contexts = entry["contexts"]
    answers_by_context = entry["answer_by_context"]

    for i, context in enumerate(contexts):
        if str(i) in answers_by_context:
            for answer_list in answers_by_context[str(i)]:
                for answer in answer_list:
                    for question in question_list:
                        entry = {
                            "question": f'Context: {context["candidate_texts"]}\n\nQuestion: {question}\nAnswer:',
                            "context": context["candidate_texts"],
                            "answer": answer,
                            # "answer_start": context["candidate_texts"].find(answer),
                            # "id": entry["q_id"]
                        }

                        hf_data.append(entry)

# Convert to Pandas DataFrame
df = pd.DataFrame(hf_data)

# Split into train (80%) and test (20%) sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Save train and test datasets separately
train_dataset.save_to_disk("datasets/whoqa_train.hf")
test_dataset.save_to_disk("datasets/whoqa_test.hf")
