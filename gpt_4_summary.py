import os
import uuid
import json

import tqdm
import openai
import tiktoken
import datasets


openai.organization = os.getenv("ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")


def make_response(prompt, inference_params):

    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": "Ты помощник в суммаризации книг."},
            {"role": "user", "content": prompt}
        ],
        **inference_params
    )

    return response["choices"][0]['message']['content']


def generate_summary(sample):
    max_tokens = 7500
    enc = tiktoken.encoding_for_model("gpt-4")
    chapters_text = sample['chapters_text']
    encoded_text = enc.encode(chapters_text)
    print(f'Current len: {len(encoded_text)}')
    if len(encoded_text) > max_tokens:
        print(f'Truncated to: {max_tokens}')
        encoded_text = encoded_text[:max_tokens]
    chapters_text = enc.decode(encoded_text)
    base_prompt = 'Напиши саммари главы книги в 4 предложениях:'
    prompt = f"{base_prompt}\n{chapters_text}\n Саммари:\n"
    inference_params = {
        'top_p': 0.95
    }
    summ = make_response(prompt, inference_params)
    return summ


if __name__ == '__main__':
    val_dataset = datasets.load_dataset('c00k1ez/summarization')['validation']

    book_id = 130
    chapters = [c for c in val_dataset if c['book_id'] == book_id]

    new_chapters = []
    for sample in tqdm.tqdm(chapters):
        summ = generate_summary(sample)
        sample['gpt_4_summary'] = summ
        new_chapters.append(sample)
    
    base_dir = './gpt_4_summary_val'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    
    with open(f'{base_dir}/gpt_4_summary_val_{uuid.uuid4()}.jsonl', 'w', encoding='utf-8') as f:
        #for c in new_chapters:
            json.dump(new_chapters, f, indent=4)
