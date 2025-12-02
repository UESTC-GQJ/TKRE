import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import json
import random
from tqdm import tqdm
import argparse
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

entity_types = {
    "tacrev": ['URL', 'LOCATION', 'IDEOLOGY', 'CRIMINAL CHARGE', 'TITLE', 'STATE OR PROVINCE', 'DATE', 'PERSON', 'NUMBER', 'CITY', 'DURATION', 'CAUSE OF DEATH', 'COUNTRY', 'NATIONALITY', 'RELIGION', 'ORGANIZATION', 'MISCELLANEOUS'],
    "retacred": ['IDEOLOGY', 'ORGANIZATION', 'URL', 'PERSON', 'DURATION', 'COUNTRY', 'LOCATION', 'NATIONALITY', 'TITLE', 'RELIGION', 'NUMBER', 'CITY', 'CAUSE OF DEATH', 'DATE', 'STATE OR PROVINCE', 'CRIMINAL CHARGE'],
    "tacred": ['COUNTRY', 'IDEOLOGY', 'LOCATION', 'DATE', 'PERSON', 'NATIONALITY', 'RELIGION', 'CITY', 'MISCELLANEOUS', 'CAUSE OF DEATH', 'TITLE', 'URL', 'NUMBER', 'ORGANIZATION', 'STATE OR PROVINCE', 'DURATION', 'CRIMINAL CHARGE']
}

def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token
    

# def model_inference(model, prompt):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     inputs = inputs.to(0)
#     generate_ids = model.generate(inputs.input_ids, do_sample=False, max_new_tokens=500)
#     output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
#     assert output[:len(prompt)] == prompt
#     # pdb.set_trace()
#     return output[len(prompt):]

def generate(_prompt, _tokenizer, _pipeline):
    response = _pipeline(
        _prompt,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=_tokenizer.eos_token_id,
        max_new_tokens=1024,
        truncation=True
    )[0]['generated_text']
    return response.split('[/INST]')[-1].strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_path', '-dp', type=str, required=True, help="The directory of demonstration data.")
    parser.add_argument('--auto_modelpath', type=str, required=True, help="The directory of LLM path.")
    parser.add_argument('--output_dir', type=str, required=True, help="The output directory of generated data.")
    parser.add_argument('--dataset', type=str, required=True, choices=["tacred", "tacrev", "retacred"])
    parser.add_argument('--k', type=int, default=3, help="k-shot demonstrations")
    args = parser.parse_args()
    
    auto_modelpath = args.auto_modelpath
    tokenizer = AutoTokenizer.from_pretrained(auto_modelpath)
    # model = AutoModelForCausalLM.from_pretrained(auto_modelpath, device_map="auto")
    pipe = pipeline(
        'text-generation',
        model='/data/guoquanjiang/Llama-2-13b-chat-hf',
        torch_dtype=torch.float16,
        device_map='auto'
    )
    input_file = args.demo_path
    datasetname = args.dataset
    output_file = args.output_dir
    data = []
    label_list = {}
    with open(input_file,'r') as f:
        data = json.load(f)
    random.shuffle(data)
    for line in data:
        rel = line['relation']
        if rel not in label_list:
            label_list[rel] = [line]
        else:
            label_list[rel].append(line)


    '''
    One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context and their entity types. 
    The head entity has the relation with the tail entity and entities are pre-categorized as the following types: URL, LOCATION, IDEOLOGY, CRIMINAL CHARGE, TITLE, STATE OR PROVINCE, DATE, PERSON, NUMBER, CITY, DURATION, CAUSE OF DEATH, COUNTRY, NATIONALITY, RELIGION, ORGANIZATION, MISCELLANEOUS. 
    Here are some samples for relation 'org:founded_by':
    Relation: org:founded_by. Context: President Lee Teng-hui confers the Order of the Brilliant Star with a Violet Grand Cordon on Samuel Noordhoff , founder of the Noordhoff Craniofacial Foundation , for his devoted service to local citizens over the past four decades. Head Entity: Noordhoff Craniofacial Foundation . Head Type: ORGANIZATION. Tail Entity: Samuel Noordhoff. Tail Type: PERSON.
    Relation: org:founded_by. Context: Talansky is also the US contact for the New Jerusalem Foundation , an organization founded by Olmert while he was Jerusalem 's mayor . Head Entity: New Jerusalem Foundation. Head Type: ORGANIZATION. Tail Entity: Olmert. Tail Type: PERSON.
    Relation: org:founded_by. Context: Sharpton has said he will not endorse any candidate until hearing more about their views on civil rights and other issues at his National Action Network convention next week in New York City . Head Entity: National Action Network. Head Type: ORGANIZATION. Tail Entity: his. Tail Type: PERSON.
    Relation: org:founded_by. Context: `` We believe that we can best serve our clients by offering a single multistrategy hedge fund platform , '' wrote John Havens , who was a founder of Old Lane with Pandit and is president of the alternative investment group . Head Entity: Old Lane. Head Type: ORGANIZATION. Tail Entity: John Havens. Tail Type: PERSON.
    Generate more samples for the relation 'org:founded_by'.
    '''

    generated_data = []

    with open(output_file, 'w') as f:
        for k, v in tqdm(label_list.items(), desc="Processing relations"):
            for i in range(args.k):
                # 构建 prompt
                prompt = '''[INST] Instruction: Take the text below and give an explanation of why the subject "{0}" has the relation "{1}" with the object "{2}". {3} [/INST]'''.format(
                    ' '.join([convert_token(token) for token in v[i]['token'][v[i]['subj_start']:v[i]['subj_end']+1]]),
                    k,
                    ' '.join([convert_token(token) for token in v[i]['token'][v[i]['obj_start']:v[i]['obj_end']+1]]),
                    ' '.join([convert_token(token) for token in v[i]['token']])
                )

                # 获取模型的响应
                response = generate(prompt, tokenizer, pipe)
                res = response.split('\n')
                res = ' '.join(res[2:])
                print(res)
                # 将每条响应保存为一个字典
                # for line in res:
                #     if len(line.strip()) == 0:
                #         continue  # 忽略空行

                    # 生成一个包含生成文本的字典，包含关系、实体、以及生成的文本
                data_entry = {
                    "context": ' '.join([convert_token(token) for token in v[i]['token']]),
                    "relation": k,
                    "subject": ' '.join([convert_token(token) for token in v[i]['token'][v[i]['subj_start']:v[i]['subj_end']+1]]),
                    "object": ' '.join([convert_token(token) for token in v[i]['token'][v[i]['obj_start']:v[i]['obj_end']+1]]),
                    "generated_text": res
                }

                # 将生成的条目加入到列表
                generated_data.append(data_entry)

        # 将所有生成的条目写入新的 JSON 文件
        json.dump(generated_data, f, ensure_ascii=False, indent=4)

    print(f"Generated data saved to {output_file}.")