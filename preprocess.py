# -*- coding: utf-8 -*-
import json
data_dir = './'
import utils
import random
import os
#from sklearn.model_selection import train_test_split
train_dir = './'
valid_dir = './'





def trans_type(file, is_train):

    if is_train == 'test':
        sentences = []
        with open(file) as reader:
            data = json.load(reader)

            for entry in data:
                text = entry['text']
                sentences.append(text)
        reader.close()
        with open(data_dir + is_train + '_sentences.txt', 'w') as writer:
            for sentence in sentences:
                writer.write(sentence + '\n')
        writer.close()
    else:
        with open(file) as reader:
            data = json.load(reader)
            sentences = []
            intents = []
            domains = []
            role_labels = []
            for entry in data:
                text = entry['text']
                text = utils.convert_to_unicode(text)
                domain = entry['domain']
                intent = entry['intent']
                slots = entry['slots']
                sentences.append(text)
                domains.append(domain)
                intents.append(intent)
                sentence_roled = sentence_role(text, slots)  # label sentence using slots
                role_labels.append(sentence_roled)
        reader.close()

        with open(data_dir + is_train + '_''sentences.txt', 'w') as writer:
            for sentence in sentences:
                writer.write(sentence + '\n')
        writer.close()
        with open(data_dir + is_train + '_' + 'intents.txt', 'w') as writer:
            for intent in intents:
                writer.write(str(intent) + '\n')
        writer.close()
        with open(data_dir + is_train + '_' + 'domains.txt', 'w') as writer:
            for domain in domains:
                writer.write(domain + '\n')
        writer.close()
        with open(data_dir + is_train + '_' + 'role_labels.txt', 'w') as writer:
            for role_label in role_labels:
                writer.write(role_label + '\n')
        writer.close()


def sentence_role(sentence, slots):
    length = len(sentence)
    role_label = ['o' for x in range(length)]
    keys = slots.keys()
    for key_idx, key in enumerate(keys):
        value = slots[key]
        postion = sentence.index(value)
        for i in range(len(value)):
            if i==0:
                role_label[postion+i] = 'B-' + key
            else:
                role_label[postion+i] = 'I-' + key

    sentence_roled =' '.join(role_label)
    return sentence_roled

'''
def split_dataset(file):
    train = []
    valid = []
    with open(data_dir + file) as reader:
        data = json.load(reader)
        data_size = len(data)
        print('data_size', data_size)
        train, valid = train_test_split(data)
    reader.close()


    with open(data_dir + 'train.json', 'w') as writer:
        writer.write(json.dumps(train, indent=4) + '\n')
    writer.close()

    with open(data_dir + 'valid.json', 'w') as writer:
        writer.write(json.dumps(valid, indent=4) + '\n')
    writer.close()
    print('split finished')
'''

def split_dataset(file, k):
    for i in range(k):
        print('genetrator %d fold data'%i)
        train = []
        valid = []
        if not os.path.exists(os.path.join(data_dir, 'data_' + str(i))):
            os.mkdir(os.path.join(data_dir, 'data_' + str(i)))
        with open(data_dir + file) as reader:
            data = json.load(reader)
            train, valid = train_test_split(data, test_size=0.2)
            print('len of train dataset', len(train))
            print('len of valid dataset', len(valid))

        reader.close()
        with open(os.path.join(data_dir, 'data_' + str(i), 'train.json'), 'w') as writer:
            writer.write(json.dumps(train, indent=4) + '\n')
        writer.close()
        with open(os.path.join(data_dir, 'data_' + str(i), 'valid.json'), 'w') as writer:
            writer.write(json.dumps(valid, indent=4) + '\n')
        writer.close()
    print('split finished')



def statics(file):
    with open(data_dir + file) as reader:
        data = json.load(reader)
        data_size = len(data)
        domain_set = set()
        intent_set = set()
        max_sen_len = 0
        slots_key_set = set()
        for entry in data:
            text = entry['text']
            domain = entry['domain']
            intent = entry['intent']
            slots = entry['slots']
            domain_set.add(domain)
            intent_set.add(intent)
            if len(text) > max_sen_len:
                max_sen_len = len(text)
            keys = set(slots.keys())
            slots_key_set = slots_key_set | keys
    reader.close()
    statics = {}
    statics.update(num_domain=len(domain_set))
    statics.update(num_intent = len(intent_set))
    statics.update(max_sen_len = max_sen_len)
    statics.update(num_slots_keys = len(slots_key_set)*2+1)
    return statics






if __name__ == '__main__':
    file = 'sample.json'
    kfold = 5
    split_dataset(file, kfold)
