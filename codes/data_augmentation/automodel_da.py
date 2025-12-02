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

def model_inference(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(0)
    generate_ids = model.generate(inputs.input_ids, do_sample=False, max_new_tokens=500)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    assert output[:len(prompt)] == prompt
    # pdb.set_trace()
    return output[len(prompt):]

def parse_generated_sample(line, relation, entity_types, datasetname):
    """
    解析大模型生成的单条样本，返回结构化数据。如果解析失败返回 None。
    """
    try:
        DAdata = {}
        data1 = line.split('Relation:')[-1].strip()
        onepoint = data1.index('.')
        relation_extracted = data1[:onepoint]
        if relation_extracted != relation:
            return None

        # 获取上下文
        data2 = data1.split('Context:')[-1].strip()
        data2lower = data2.lower()
        if "head entity:" in data2lower:
            textend = data2lower.index('head entity:')
            text = data2[:textend].strip()
            data3 = data2[textend + len('head entity:'):].strip()
        else:
            return None

        DAdata['text'] = text

        # 获取 Head Entity
        data3lower = data3.lower()
        if ". head type:" in data3lower:
            headend = data3lower.index(". head type:")
            head = data3[:headend].strip()
            data4 = data3[headend + len(". head type:"):].strip()
        else:
            return None

        # 获取 Head Type
        data4lower = data4.lower()
        if ". tail entity:" in data4lower:
            htend = data4lower.index(". tail entity:")
            headtype = data4[:htend].strip()
            if headtype not in entity_types[datasetname]:
                return None
            data5 = data4[htend + len(". tail entity:"):].strip()
            DAdata['subj_type'] = format_entity_type(headtype, datasetname)
        else:
            return None

        # 获取 Tail Entity
        data5lower = data5.lower()
        if ". tail type:" in data5lower:
            tailend = data5lower.index(". tail type:")
            tail = data5[:tailend].strip()
            data6 = data5[tailend + len(". tail type:"):].strip()
        else:
            return None

        # 获取 Tail Type
        tailtype = data6[:-1].strip()
        if tailtype not in entity_types[datasetname]:
            return None
        DAdata['obj_type'] = format_entity_type(tailtype, datasetname)

        # 获取实体位置
        textlower = text.lower()
        headlower = head.lower()
        if headlower in textlower:
            hpos1 = textlower.index(headlower)
            hpos2 = hpos1 + len(headlower)
            DAdata['subj'] = text[hpos1:hpos2]
            DAdata['subj_start'], DAdata['subj_end'] = hpos1, hpos2
        else:
            return None

        taillower = tail.lower()
        if taillower in textlower:
            tpos1 = textlower.index(taillower)
            tpos2 = tpos1 + len(taillower)
            DAdata['obj'] = text[tpos1:tpos2]
            DAdata['obj_start'], DAdata['obj_end'] = tpos1, tpos2
        else:
            return None

        DAdata['relation'] = relation
        return DAdata
    except:
        return None

def format_entity_type(entity_type, datasetname):
    """
    格式化实体类型，适配不同数据集的命名规范。
    """
    if datasetname in ["tacrev", "tacred", "retacred"]:
        entity_type = entity_type.upper()
        if entity_type == "MISCELLANEOUS":
            entity_type = "MISC"
        return entity_type.replace(" ", "_")
    elif datasetname == "SciERC":
        return entity_type.title()
    return entity_type

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
        model='/home/guoquanjiang/Chinese-LLaMA-Alpaca-2/models/ydyajyA/Llama-2-13b-chat-hf',
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

    with open(output_file, 'w') as f:
        for k, v in tqdm(label_list.items(), desc="Generating samples"):
            generated_samples = []
            generated_texts = set()  # 记录已生成的文本，防止重复
            max_attempts = 10  # 最大生成尝试次数
            
            while len(generated_samples) < 8:
                # 构建 Prompt
                prompt = '''
                    "One sample in relation extraction datasets consists of a relation, "
                    "a context, a pair of head and tail entities in the context and their entity types. "
                    "The head entity has the relation with the tail entity and entities are pre-categorized as "
                    "the following types: " {0} ". Here are some samples for relation '" {1} "':\n"
                '''.format(', '.join(entity_types[datasetname]),k)
                for i in range(min(len(v), args.k)):  # 使用原始样本填充 Prompt
                    sample = (
                        "Relation: " + k + ". Context: " + ' '.join([convert_token(token) for token in v[i]['token']]) + 
                        " Head Entity: " + ' '.join([convert_token(token) for token in v[i]['token'][v[i]['subj_start']:v[i]['subj_end']+1]]) + 
                        ". Head Type: " + v[i]['subj_type'] + 
                        ". Tail Entity: " + ' '.join([convert_token(token) for token in v[i]['token'][v[i]['obj_start']:v[i]['obj_end']+1]]) + 
                        ". Tail Type: " + v[i]['obj_type'] + ".\n"
                    )
                    prompt += sample
                
                prompt += "Generate eight more samples like above for the relation '" + k + "':"
                
                # 调用模型生成
                try:
                    response = generate(prompt, tokenizer, pipe)
                    res = response.split('\n')
                except Exception as e:
                    print(f"Error generating samples for relation '{k}': {e}")
                    continue
                
                for line in res:
                    if not line.strip():
                        continue
                    try:
                        DAdata = parse_generated_sample(line, k, entity_types, datasetname)
                        if DAdata is not None and DAdata['text'] not in generated_texts:
                            generated_texts.add(DAdata['text'])
                            generated_samples.append(DAdata)
                            if len(generated_samples) == 8:
                                break
                    except Exception as e:
                        # 捕获单条样本处理错误，不影响其他样本
                        print(f"Error parsing line: {line}. Error: {e}")
            
            # 写入生成的样本到文件
            for sample in generated_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

