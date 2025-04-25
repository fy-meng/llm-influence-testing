from openai import OpenAI
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

OPENAI_API_KEY = ('sk-proj-PvArJfBA82KsMFtAay6n8UneEceRpFsAPgYwSOFkx8uegaPQtxkGw'
                  'whJZtQvUQQKHW5CVzPJ1kT3BlbkFJjEAlg-2djav_Jasf4_LvSNoQebO2GnYk'
                  'FLXru8SOPi-KDMQtKNUX_ieFJFzEvZXHNABmXD8EIA')

prompts = {
    'question_cf_strong': "You will be given a question-answer pair. Find the "
                          "subject on the question, and change it to a different "
                          "noun, while maintaining the question's fluidity. Only "
                          "output the question with the modified subject. \n"
                          "Question: {question} Answer: {answer}",
    'answer_cf_strong': "You will be given a question-answer pair. Find the "
                        "subject on the answer, and change it to a different "
                        "noun, while maintaining the answer's fluidity. Only "
                        "output the answer with the modified subject. \n"
                        "Question: {question} Answer: {answer}",
    'question_cf_weak': "You will be given a question-answer pair. Change one "
                        "word or phrase in the question without changing the "
                        "meaning of the question, while maintaining the question's "
                        "fluidity. Only output the modified question. \n"
                        "Question: {question} Answer: {answer}",
    'answer_cf_weak': "You will be given a question-answer pair. Change one "
                      "word or phrase in the answer without changing the "
                      "meaning of the answer, while maintaining the answer's "
                      "fluidity. Only output the modified answer. \n"
                      "Question: {question} Answer: {answer}",
}

client = OpenAI(api_key=OPENAI_API_KEY)

dataset = load_dataset('wiki_qa')

np.random.seed(42)

train_size = 100
train_idx = np.random.choice(np.arange(len(dataset['train'])), train_size, replace=False)
test_size = 100
test_idx = np.random.choice(np.arange(len(dataset['test'])), test_size, replace=False)
indices = {'train': train_idx, 'test': test_idx}

for prompt_tag, prompt in prompts.items():
    print(f'generating cf for {prompt_tag}')
    modification = prompt_tag.split('_')[0]
    dataset = load_dataset('wiki_qa')  # reload dataset


    def counterfactual(example):
        response = client.responses.create(
            model='gpt-4.1',
            input=prompt.format(question=example['question'], answer=example['answer']),
        )
        example[modification] = response.output_text
        example['prompt'] = f"Question: {example['question']} Answer:"
        return example


    for split in ['train', 'test']:
        dataset_cf = dataset[split].select(indices[split])
        dataset_cf = dataset_cf.map(counterfactual)
        dataset_cf.save_to_disk(f'datasets/wikiqa_{prompt_tag}_{split}.hf')
