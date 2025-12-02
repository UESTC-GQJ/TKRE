import pdb
import json
import os

def str_list_compare(s_list1, s_list2):
    if len(s_list1) != len(s_list2):
        return False
    for idx in range(len(s_list1)):
        if s_list1[idx] != s_list2[idx]:
            return False
    return True

def process_token(text, textlower, headlower, taillower, hpos1, tpos1):
    token_origin = text.split(' ')
    token = textlower.split(' ')
    
    if not str_list_compare(token, [t.lower() for t in token_origin]):
        print("Tokenization mismatch.")
        return [], -1, -1, -1, -1
    
    if hpos1 == 0:
        ss = 0
    else:
        num = 0
        ss = -1
        for idx in range(len(token)):
            word = token[idx]
            num += len(word) + 1
            if num == hpos1:
                ss = idx
                break
        if ss == -1:
            print(f"hpos1 {hpos1} not found in textlower.")
            return [], -1, -1, -1, -1
    
    se = ss + len(headlower.split(' ')) - 1
    
    if tpos1 == 0:
        os = 0
    else:
        num = 0
        os = -1
        for idx in range(len(token)):
            word = token[idx]
            num += len(word) + 1
            if num == tpos1:
                os = idx
                break
        if os == -1:
            print(f"tpos1 {tpos1} not found in textlower.")
            return [], -1, -1, -1, -1
    
    oe = os + len(taillower.split(' ')) - 1
    return token_origin, ss, se, os, oe

def process_line(DAdata):
    required_keys = ['text', 'subj_start', 'subj_end', 'subj_type', 'obj_start', 'obj_end', 'obj_type', 'relation']
    if not all(key in DAdata for key in required_keys):
        print(f"Missing keys in DAdata: {DAdata}")
        return False, {}
    
    text = DAdata['text']
    truehead = DAdata['subj']
    hpos1, hpos2 = DAdata['subj_start'], DAdata['subj_end']
    truetail = DAdata['obj']
    tpos1, tpos2 = DAdata['obj_start'], DAdata['obj_end']
    relation = DAdata['relation']
    
    textlower = text.lower()
    headlower = truehead.lower()
    taillower = truetail.lower()
    
    try:
        token_origin, ss, se, os, oe = process_token(text, textlower, headlower, taillower, hpos1, tpos1)
        if ss == -1 or os == -1:
            return False, {}
    except Exception as e:
        print(f"Exception in process_line: {e}")
        return False, {}
    
    processed_DAdata = {
        'text': text,
        'token': token_origin,
        'subj_start': ss,
        'subj_end': se,
        'subj_type': DAdata['subj_type'],
        'obj_start': os,
        'obj_end': oe,
        'obj_type': DAdata['obj_type'],
        'relation': relation,
        'ly_headlower': headlower,
        'ly_taillower': taillower
    }
    return True, processed_DAdata

def merge(origin_filepath, da_filepath_list, output_filepath, add_k=8):
    try:
        with open(origin_filepath, 'r') as f_in:
            data = json.load(f_in)
    except Exception as e:
        print(f"Error loading original data: {e}")
        return
    
    da_data = {}
    for line in data:
        if line['relation'] not in da_data:
            da_data[line['relation']] = []
    
    count = 0
    for da_filepath in da_filepath_list:
        try:
            with open(da_filepath, 'r') as f_in:
                for line in f_in:
                    try:
                        line_json = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Failed to parse line: {line}")
                        continue
                    signal, processed_line = process_line(line_json)
                    if signal:
                        relation = processed_line['relation']
                        if relation not in da_data:
                            da_data[relation] = []
                        da_data[relation].append(processed_line)
                        count += 1
        except FileNotFoundError:
            print(f"File not found: {da_filepath}")
            continue
    print(f"Total augmented examples loaded: {count}")
    
    for relation in da_data:
        print(f"Relation: {relation}, Number of augmented examples: {len(da_data[relation])}")
        if len(da_data[relation]) < add_k:
            print(f"Warning: Relation {relation} has fewer than {add_k} examples.")
        if relation in da_data and isinstance(da_data[relation], list):
            data.extend(da_data[relation][:add_k])
        else:
            print(f"Warning: No augmented data for relation {relation}")
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w') as f_out:
        json.dump(data, f_out, indent=True)

if __name__ == '__main__':
    merge(
        origin_filepath='/data/guoquanjiang/GenPT/data/tacred/aug-k-shot/8-21/train.json',
        da_filepath_list=['/data/guoquanjiang/DSARE/datasets/llama2_13B_da.json'],
        output_filepath='../../datasets/example_data/merged_train.json',
        add_k=8
    )

