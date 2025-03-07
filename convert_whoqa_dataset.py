import json
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the JSON file
with open("WhoQA/WhoQA.json", "r", encoding="utf-8") as file:
    data = json.load(file)

print('len', len(data))
data = data[:len(data) // 100]
print('new len', len(data))

# Prepare a list for Hugging Face dataset format
hf_data_train = []
hf_data_test = []

for entry in tqdm(data):
    question_list = entry["questions"]
    contexts = entry["contexts"]
    answers_by_context = entry["answer_by_context"]

    for i, context in enumerate(contexts):
        if str(i) in answers_by_context:
            for answer_list in answers_by_context[str(i)]:
                for answer in answer_list:
                    for question in question_list:
                        entry_train = {
                            "question": f'Question: {question}\nContext: {context["candidate_texts"]}\nAnswer: ',
                            "context": context["candidate_texts"],
                            "answer": answer
                        }
                        entry_test = {
                            "question": f'Question: {question}\nAnswer: ',
                            "context": '',
                            "answer": answer
                        }

                        hf_data_train.append(entry_train)
                        hf_data_test.append(entry_test)

# Convert to Pandas DataFrame
df_train = pd.DataFrame(hf_data_train)
df_test = pd.DataFrame(hf_data_test)

# Split into train (80%) and test (20%) sets
train_df, _test_df = train_test_split(df_train, test_size=0.4, random_state=42)
eval_df, test_df = train_test_split(_test_df, test_size=0.5, random_state=42)

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
test_dataset = Dataset.from_pandas(test_df)

# Save train and test datasets separately
train_dataset.save_to_disk("datasets/whoqa_train.hf")
eval_dataset.save_to_disk("datasets/whoqa_eval.hf")
test_dataset.save_to_disk("datasets/whoqa_test.hf")
