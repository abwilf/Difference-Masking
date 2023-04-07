import tqdm
from transformers import BertTokenizer, BertModel, BertTokenizerFast, RobertaModel, RobertaTokenizerFast
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

tokenizer_bert = BertTokenizerFast.from_pretrained("bert-base-uncased")
model_bert = BertModel.from_pretrained("bert-base-uncased")
tokenizer_roberta_fast = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
model_roberta = RobertaModel.from_pretrained("roberta-base")

def get_seed_emb(data_type, method):
    if data_type == 'chem' and method == 'expert':
        seeds = ['chemistry', 'inhibitor', 'cells', 'acid', 'receptor', 'protein', 'kinase', 'phosphorylation', 'antagonist', 'enzyme', 'activity', 'mrna', 'synthase', 'tyrosine', 'agonist','synthesis', 'binding', 'apoptosis', 'acetylcholinesterase', 'cytochrome']
    elif data_type == 'chem' and method == 'tf_idf':
        seeds = ['inos','adrenoceptor','inhibit','tcdd','akt','agonist', 'vegf', 'caspase', 'inhibiting', 'ec','cyclin', 'microm', 'antagonist', 'potency', 'antagonists', 'inhibitory', 'potent', 'inhibits', 'tyrosine', 'synthase']
    elif data_type == 'acl' and method == 'expert':
        seeds = ['language', 'text', 'model', 'information', 'grammar', 'lexical', 'translation', 'learning', 'word', 'corpus', 'semantic', 'syntactic', 'features', 'structure', 'data', 'training', 'rules', 'analysis', 'parser', 'statistical']
    else:
        seeds = ['charniak', 'coreference', 'treebank', 'parsers', 'grammars', 'parse', 'syntactic', 'linguistic', 'lexical', 'roth', 'och', 'parser', 'parsing', 'extraction', 'chen', 'sentences', 'phrase', 'semantic', 'sentence', 'dialogue']
    input_chem = tokenizer_bert.batch_encode_plus(seeds, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)
    # input_chem = tokenizer_bert.batch_encode_plus(['language', 'text', 'model', 'information', 'grammar', 'lexical'], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)
    output_chem = model_bert(**input_chem)
    return output_chem

def get_similarity(sents, data_type, method):
    if data_type == 'chem' and method == 'expert':
        seeds = ['chemistry', 'inhibitor', 'cells', 'acid', 'receptor', 'protein', 'kinase', 'phosphorylation', 'antagonist', 'enzyme', 'activity', 'mrna', 'synthase', 'tyrosine', 'agonist','synthesis', 'binding', 'apoptosis', 'acetylcholinesterase', 'cytochrome']
    elif data_type == 'chem' and method == 'tf_idf':
        seeds = ['inos','adrenoceptor','inhibit','tcdd','akt','agonist', 'vegf', 'caspase', 'inhibiting', 'ec','cyclin', 'microm', 'antagonist', 'potency', 'antagonists', 'inhibitory', 'potent', 'inhibits', 'tyrosine', 'synthase']
    elif data_type == 'acl' and method == 'expert':
        seeds = ['language', 'text', 'model', 'information', 'grammar', 'lexical', 'translation', 'learning', 'word', 'corpus', 'semantic', 'syntactic', 'features', 'structure', 'data', 'training', 'rules', 'analysis', 'parser', 'statistical']
    else:
        seeds = ['charniak', 'coreference', 'treebank', 'parsers', 'grammars', 'parse', 'syntactic', 'linguistic', 'lexical', 'roth', 'och', 'parser', 'parsing', 'extraction', 'chen', 'sentences', 'phrase', 'semantic', 'sentence', 'dialogue']
        
        
    clean_sent = [word.strip('.,') for sent in sents for word in sent.lower().strip('.!').split(" ")]
    input_cytokine = tokenizer_bert.batch_encode_plus(clean_sent, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)
    output_cytokine = model_bert(**input_cytokine)
    sim = []
    which_seed = []
    for idx in tqdm.tqdm(range(output_cytokine.pooler_output.shape[0])):
        if emb_chem.pooler_output.shape[0] > 1:
            temp_sim = -1000.00
            tseed = ''
            for sidx in range(emb_chem.pooler_output.shape[0]):
                tsim = torch.cosine_similarity(emb_chem.pooler_output[sidx][None], output_cytokine.pooler_output[idx][None]).detach().numpy()[0]
                if temp_sim < tsim:
                    temp_sim = tsim
                    tseed = seeds[sidx]
            sim.append(temp_sim)
            which_seed.append(tseed)
        else:
            sim.append(torch.cosine_similarity(emb_chem.pooler_output, output_cytokine.pooler_output[idx][None]).detach().numpy()[0])
    sim = np.array(sim)
    sim = np.divide(sim, (1.0+np.exp(-sim)))
    sim /= sim.sum()
    K = int(0.50*len(sim)) #int(0.15*len(sim))
    #indices = np.argpartition(sim,-K)[-K:]
    indices = (-sim).argsort()[:K]
    words = []
    for sent in sents:
        words.extend(sent.split(" "))
    sim_word = [words[idx] for idx in indices]
    return sim_word, which_seed
    
    
import json
import argparse
from collections import Counter, defaultdict
# docs = json.load(open('/work/sakter/AANG/datasets/chemprot/train.jsonl', 'r'))

parser = argparse.ArgumentParser("simple_example")
parser.add_argument("--dtype", help="An integer will be increased by 1 and printed.", type=str)
parser.add_argument("--method", help="An integer will be increased by 1 and printed.", type=str)
args = parser.parse_args()


dtype = args.dtype
method = args.method
print(dtype, method)

emb_chem = get_seed_emb(dtype, method)

if dtype == 'acl':
    filename = '/work/sakter/AANG/datasets/citation_intent/train.txt'
else:
    filename = '/work/sakter/AANG/datasets/chemprot/train.txt'
    
with open(filename, encoding="utf-8") as f:
    token_counter = Counter()
    lines = []
    all_tokens = []
    capitalizations = []
    for line in f.readlines():
        line = line.strip()
        if len(line) < 2: # Remove all single letter or empty lines
            continue
        lines.append(line)
        
        
all_sim_word = []
all_seed_word = []
for i in range(0, len(lines), 20):
    # get the sublist of the next `n` elements
    sublist = lines[i:i+20]
    # do something with the sublist
    sim_word, which_seed = get_similarity(sublist, dtype, method)
    all_sim_word.extend(sim_word)
    all_seed_word.extend(which_seed)
    
x = Counter(all_sim_word)
x = dict(x)
import json
with open(f'data_{dtype}_{method}.json', 'w') as fp:
    json.dump(x, fp)
    
x = Counter(all_seed_word)
x = dict(x)
with open(f'data_{dtype}_seed_{method}.json', 'w') as fp:
    json.dump(x, fp)