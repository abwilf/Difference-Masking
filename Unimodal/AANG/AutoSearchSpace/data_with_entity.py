from nltk import tokenize as nltktokenize
import os
import logging
import json
import unicodedata
import numpy as np
from collections import Counter, defaultdict
from data_utils import *
import math
import pdb
import random
from Bio_Epidemiology_NER.bio_recognizer import ner_prediction

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
logger = logging.getLogger(__name__)

import sys
PATH = os.path.join(os.getcwd(), "dont_stop_pretraining/")
sys.path.insert(1, PATH)
from dataset.dataset_readers.text_classification_json_reader_with_sampling import TextClassificationJsonReaderWithSampling


OUT_PAD = -100

def add_data_args(parser):
	parser.add_argument('-task-data', type=str, default=None)
	parser.add_argument('-in-domain-data', type=str, default=None)
	parser.add_argument('-out-domain-data', type=str, default=None)
	parser.add_argument('-neural-lm-data', type=str, default=None)


DATA_PATHS = {
	'CITATION_INTENT': '/home/ldery/internship/dsp/datasets/citation_intent/train.jsonl',
	'CHEMPROT': '/work/sakter/AANG/temp_datasets/chemprot/train.jsonl',
	'SCIIE':  '/home/ldery/internship/dsp/datasets/sciie/train.jsonl',
	'HYPERPARTISAN':  '/home/ldery/internship/dsp/datasets/hyperpartisan/train.jsonl',
	'SemEval2016Task6':  '/home/ldery/internship/dsp/datasets/SemEval2016Task6/train.jsonl',
	'PERSPECTRUM':'/home/ldery/internship/dsp/datasets/PERSPECTRUM/train.jsonl',
}

import pdb
class SupervisedDataset(Dataset):
	def __init__(self, dataset_name):
		self.is_supervised = True
		self.process_dataset(dataset_name)

	def process_dataset(self, dataset_name):
		dataset_reader = TextClassificationJsonReaderWithSampling(
							token_indexers=None, tokenizer=None,
							max_sequence_length=None, lazy=None
						)
		# Read from the dataset
		fname = DATA_PATHS[dataset_name]
		all_samples = dataset_reader._read(fname, raw_text=True)
		self.examples = []
		all_labels = []
		for instance in all_samples:
			text, label = instance
			if label not in all_labels:
				all_labels.append(label)
			label_idx = all_labels.index(label)
			self.examples.append({'text': text, 'label': label_idx})


	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i):
		this_sample = self.examples[i]
		return {'sample': this_sample['text'], 'label':this_sample['label'], 'idx': i}

	def get_samples(self, n_samples, is_sent_config=False):
		total_samples = len(self.examples)
		chosen_idxs = np.random.choice(total_samples, n_samples)
		all_chosen = [self[idx] for idx in chosen_idxs]
		return all_chosen


class MNLIDataset(Dataset):
	def __init__(self, tokenizer):
		dataset = load_dataset('multi_nli', split='train')
		self.tokenizer = tokenizer
		self.is_supervised = True
		self.process_dataset(dataset)
	
	def process_dataset(self, dataset):
		self.examples = []
		for entry in dataset:
			this_entry = {
				'label': int(entry['label']),
				'premise': entry['premise'],
				'hypothesis':  entry['hypothesis'],
			}
			self.examples.append(this_entry)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i):
		this_sample = self.examples[i]
		return {'sample': (this_sample['premise'], this_sample['hypothesis']), 'label':this_sample['label'], 'idx': i}

	def get_samples(self, n_samples, is_sent_config=False):
		total_samples = len(self.examples)
		chosen_idxs = np.random.choice(total_samples, n_samples)
		all_chosen = [self[idx] for idx in chosen_idxs]
		return all_chosen


class LineByLineRawTextDataset(Dataset):
	def __init__(self, file_path, tokenizer, tf_or_idf_present, cap_present, max_=10):
		assert os.path.isfile(file_path)
		logger.info("Creating Raw Line x Line Dataset %s", file_path)
		
		all_lines = []
		all_tokens = []
		self.doc_lens = []
		self.doc_names = []
		self.max_ = max_
		self.is_supervised = False

		if cap_present:
			all_caps = []
		if tf_or_idf_present:
			doc_tfs = Counter()
			self.doc_tfs = {}
	
		if '.json' in file_path:
			docs = json.load(open(file_path, 'r'))
		else:
			docs = {'0': file_path}
		# Process each document
		self.doc_lens.append(0)
		for doc_id, file_path in docs.items():
			self.doc_names.append(doc_id)
			# Get the lines, token frequencies and capitalization
			freq_cntr, lines, tokens, caps = self.process_single_file(file_path, tokenizer, tf_or_idf_present, cap_present)
			all_tokens.extend(tokens)
			all_lines.extend(lines)
			self.doc_lens.append(len(lines))
			if cap_present:
				all_caps.extend(caps)
			if tf_or_idf_present:
				tf_info = scale(freq_cntr, max_=self.max_, smoothfactor=1)
				keys, values = list(tf_info.keys()), np.ceil(np.log10(list(tf_info.values())))
				# get the minimum value
				to_add = min(values) * np.sign(min(values)) + 1
				values = values + to_add
				tf_info = defaultdict(int)
				for k, v in zip(keys, values):
					tf_info[k] = v if v < 3 else 3  # This is currently hard-coded to ensure that we have 7 classes. Need to fix
				self.doc_tfs[doc_id] = tf_info
				doc_tfs.update(list(freq_cntr.keys()))

		self.examples = all_lines
		self.tokens = all_tokens
		self.doc_lens = np.cumsum(self.doc_lens)
		if tf_or_idf_present:
			# need to do some idf computation here
			smoothed_n = 1 + len(docs)
			doc_idfs = {x : np.log(smoothed_n/(1 + v) + 1.0) for x, v in doc_tfs.items()}
			self.doc_tfidfs = self.get_tfidfs(self.doc_tfs, doc_idfs)
		if cap_present:
			self.caps = all_caps
			assert len(self.caps) == len(self.examples), 'The number of caps should match the number of example sentences'


	def get_tfidfs(self, doc_tfs, doc_idfs):
		doc_tfidfs = defaultdict(lambda:defaultdict(float))
		for k, tf in doc_tfs.items():
			new_tfidf = {x: v * doc_idfs[x] for x, v in tf.items()}
			doc_tfidfs[k] = scale(new_tfidf, smoothfactor=1)
		return doc_tfidfs

	def process_single_file(self, file_path, tokenizer, tf_or_idf_present, cap_present):
		with open(file_path, encoding="utf-8") as f:
			token_counter = Counter() if tf_or_idf_present else None
			lines = []
			all_tokens = []
			capitalizations = [] if cap_present else None
			for line in f.readlines():
				line = line.strip()
				if len(line) < 2: # Remove all single letter or empty lines
					continue
				lines.append(line)
				tokens = tokenizer.tokenize(line)
				if cap_present:
					caps = get_caps(run_strip_accents(line), tokens)
					assert len(tokens) == len(caps)
					capitalizations.append(caps)
				if tf_or_idf_present:
					token_counter.update(tokens)
				all_tokens.append(tokens)
		return token_counter, lines, all_tokens, capitalizations

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i):
		return {'sample': self.examples[i], 'idx': i}

	def get_samples(self, n_samples, is_sent_config=False):
		total_samples = len(self.examples)
		if is_sent_config:
			n_samples = n_samples // 2
			total_samples -= 1
		chosen_idxs = np.random.choice(total_samples, n_samples)
		all_chosen = [self[idx] for idx in chosen_idxs]
		if is_sent_config:
			second_half = [self[idx + 1] for idx in chosen_idxs]
			all_chosen.extend(second_half)
		return all_chosen

	def getdocid(self, sent_idx):
		idx = np.searchsorted(self.doc_lens, sent_idx, side="left")
		if self.doc_lens[idx] > sent_idx:
			return self.doc_names[idx - 1]
		return self.doc_names[idx]

	def _pad_specials(self, sent_, special_tok_mask):
		if len(sent_) == len(special_tok_mask):
			assert sum(special_tok_mask) > 0, 'Sentence and token mask same length but there are special tokens'
			return torch.tensor(sent_)

		assert len(sent_) < len(special_tok_mask), 'Sentence must have fewer tokens than the special tokens mask'
		mul_ = 1.0 if isinstance(sent_[0], float) else 1
		new_sent_ = torch.full((len(special_tok_mask), ), OUT_PAD * mul_) # We only compute loss on masked tokens
		for idx_, val_ in enumerate(sent_):
			new_sent_[idx_ + 1] = val_ # plus 1 because we are skipping the bos token
		return new_sent_

	def getcaps(self, sent_idx, special_token_mask):
		return self._pad_specials(self.caps[sent_idx], special_token_mask)

	def gettfs(self, sent_idx, special_token_mask):
		sent_ = self.tokens[sent_idx]
		tf_cntr = self.doc_tfs[self.getdocid(sent_idx)]
		tfs = [tf_cntr[x] for x in sent_]
		return self._pad_specials(tfs, special_token_mask)

	def gettfidfs(self, sent_idx, special_token_mask):
		sent_ = self.tokens[sent_idx]
		tfidf_cntr = self.doc_tfidfs[self.getdocid(sent_idx)]
		tfidfs = [tfidf_cntr[x] for x in sent_]
		return self._pad_specials(tfidfs, special_token_mask)

class DataOptions(object):
	def __init__(self, args, tokenizer, bert_tokenizer, bert_model, data_dict, output_dict):
		self.construct_dataset_map(args, data_dict, output_dict, tokenizer)
		self.tokenizer = tokenizer
		self.tokenizer_bert = bert_tokenizer
		self.model_bert = bert_model
		self.max_ = 10 # This is a magic number showing the max scale for TFIDF style losses

	def construct_dataset_map(self, args, data_dict, output_dict, tokenizer):
		self.id_to_dataset_dict = {}
		tf_or_idf_present = ('TFIDF' in output_dict.values()) or ('TF' in output_dict.values())
		cap_present = 'CAP' in output_dict.values()

		for v, k in data_dict.items():
			path = None
			if k == 'Task':
				assert args.task_data is not None, 'Task Data Location not specified'
				path = args.task_data
			elif k == 'In-Domain':
				assert args.in_domain_data is not None, 'In Domain Data Location not specified'
				path = args.in_domain_data
			elif k == 'Out-Domain':
				assert args.out_domain_data is not None, 'In Domain Data Location not specified'
				path = args.out_domain_data
			elif k == 'Neural-LM':
				assert args.neural_lm_data is not None, 'In Domain Data Location not specified'
				path = args.neural_lm_data

			if k == 'MNLI':
				dataset = MNLIDataset(tokenizer)
			elif k in list(DATA_PATHS.keys()):
				dataset = SupervisedDataset(k)
			else:
				assert path is not None, 'Invalid data type given. {} : {}'.format(k, v)
				dataset = LineByLineRawTextDataset(path, tokenizer, tf_or_idf_present, cap_present)
			self.id_to_dataset_dict[v] = dataset

	def get_dataset(self, id_):
		return self.id_to_dataset_dict[id_]

	def get_total_len(self):
		lens = [len(ds) for id_, ds in self.id_to_dataset_dict.items()]
		return sum(lens)

	def get_dataset_len(self, id_):
		return len(self.id_to_dataset_dict[id_])


class DataTransformAndItr(object):
	def __init__(self, args, dataoptions, input_tform_dict, output_dict):
		# Sets the total batch size
		self.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
		self.dataOpts = dataoptions
		self.input_tform_dict = input_tform_dict
		self.output_dict = output_dict
		self.proba = args.mlm_probability
		self.block_size = args.block_size
		self.special_token_list = dataoptions.tokenizer.all_special_ids
		self.special_token_list.remove(dataoptions.tokenizer.mask_token_id)
		self.special_token_list.remove(dataoptions.tokenizer.unk_token_id)
		self.prim_task_id = args.prim_task_id
		self.num_seed = args.num_seed
		self.emb_chem = self.get_seed_emb()  
		self.sim_dict = {}
		self.mask_selection_process = args.mask_selection_process
		self.alpha = args.alpha
		self.has_alpha = args.has_alpha
    

	def total_iters(self):
		return int(self.dataOpts.get_total_len() / self.train_batch_size)

	def apply_in_tform(self, masked_tok_indices, sent, token_probas=None):
		return mask_tokens(sent, self.dataOpts.tokenizer, self.proba, token_probas, masked_tok_indices=masked_tok_indices)

	def apply_out_tform(
							self, output_type, ds, padded_sent, tformed_sent,
							orig_samples, is_supervised=False
						):
		if output_type == 'DENOISE':
			assert padded_sent.shape == tformed_sent.shape, 'Invalid Shapes. Input must have same shape as output'
			return {'input': padded_sent, 'output': tformed_sent}
		elif is_supervised:
			assert padded_sent.shape[0] == tformed_sent.shape[0], 'Invalid Shapes. Must have same batch size for input and output in supervised setting'
			return {'input': padded_sent, 'output': tformed_sent}
		elif output_type == 'TFIDF':
			tfidf_sent = [ds.gettfidfs(x['idx'], special_tok_mask[id_]) for id_, x in enumerate(orig_samples)]
			tfidf_sent = pad_sequence(tfidf_sent, OUT_PAD)
			assert padded_sent.shape == tfidf_sent.shape, 'Invalid Shapes. Input must have same shape as output'
			return {'input': padded_sent, 'output': tfidf_sent}
		elif output_type == 'TF':
			special_tok_mask = sum([padded_sent == x for x in self.special_token_list])
			special_tok_mask = (special_tok_mask > 0) * 1
			tf_sent = []
			for id_, x in enumerate(orig_samples):
				this_tf_sent = ds.gettfs(x['idx'], special_tok_mask[id_])
				tf_sent.append(this_tf_sent)
			tf_sent = torch.stack(tf_sent).to(padded_sent.device)
			assert padded_sent.shape == tf_sent.shape, 'Invalid Shapes. Input must have same shape as output'
			return {'input': padded_sent, 'output': tf_sent}
		elif output_type == 'CAP':
			cap_sent = [ds.getcaps(x['idx'], special_tok_mask[id_]) for id_, x in enumerate(orig_samples)]
			cap_sent = pad_sequence(cap_sent, OUT_PAD)
			assert padded_sent.shape == cap_sent.shape, 'Invalid Shapes. Input must have same shape as output'
			return {'input': padded_sent, 'output': cap_sent}
		elif  output_type == 'NSP':
			# Todo[ldery] - implement
			raise NotImplementedError('This output type [{}] has not been impemented yet'.format(output_type)) 
		elif  output_type == 'QT':
			# Todo[ldery] - implement
			raise NotImplementedError('This output type [{}] has not been impemented yet'.format(output_type)) 
		elif  output_type == 'FS':
			assert padded_sent.shape == tformed_sent.shape, 'Invalid Shapes. Input must have same shape as output'
			return {'input': padded_sent, 'output': tformed_sent}
		elif  output_type == 'ASP':
			# Todo[ldery] - implement
			raise NotImplementedError('This output type [{}] has not been impemented yet'.format(output_type)) 
		elif  output_type == 'SO':
			# Todo[ldery] - implement
			raise NotImplementedError('This output type [{}] has not been impemented yet'.format(output_type)) 
		elif  output_type == 'SO':
			# Todo[ldery] - implement
			raise NotImplementedError('This output type [{}] has not been impemented yet'.format(output_type)) 
		elif  output_type == 'SCP':
			# Todo[ldery] - implement
			raise NotImplementedError('This output type [{}] has not been impemented yet'.format(output_type)) 
		else:
			raise ValueError('Illegal value for output transform : {}'.format(self.output_type))

	def get_data(self, sample_configs, searchOpts, rep_tform):
		aggregated_data = []
		# compute config marginals
		rep_tforms = np.unique([config[2] for config in sample_configs])
		rep_probas = searchOpts.get_relative_probas(2, rep_tforms)

		for rep_idx, rep_id in enumerate(rep_tforms):
			try:
				num_samples = math.ceil(self.train_batch_size * rep_probas[rep_idx].item())
			except:
				assert False, 'There has been an issue calculating num_samples. Probably because rep_proba has NaN in it'

			this_rep_configs = [config for config in sample_configs if config[2] == rep_id]
			datasets = np.unique([config[0] for config in this_rep_configs])
			ds_probas = searchOpts.get_relative_probas(0, datasets)

			for ds_idx, ds_id in enumerate(datasets):
				n_ds_samples = math.ceil(num_samples * ds_probas[ds_idx].item())
				if n_ds_samples < 1:
					continue
				ds = self.dataOpts.get_dataset(ds_id)
				this_ds_configs = [config for config in this_rep_configs if config[0] == ds_id]
				out_tforms = np.unique([config[-1] for config in this_ds_configs])
				is_sent_config = np.any([searchOpts.config.is_dot_prod(x) for x in out_tforms])
				examples = ds.get_samples(n_ds_samples, is_sent_config=is_sent_config)
				# save_similarity(examples)                

				token_tforms = np.unique([config[1] for config in this_ds_configs])
				probas = searchOpts.get_relative_probas(1, token_tforms)

				_, stage_map = searchOpts.config.get_stage_w_name(1)
				tform_names = [stage_map[x] for x in token_tforms]

				for t_idx, (t_name, tform) in enumerate(zip(tform_names, token_tforms)):
					tform_configs = [config for config in this_ds_configs if config[1] == tform]
					# Get the number of samples corresponding to this transform
					this_egs = examples
					if probas[t_idx] < 1:
						this_sz = math.ceil(len(examples)*probas[t_idx])
						if this_sz < 1: # We are skipping this loss because it has been critically downweighted
							continue
						this_egs = np.random.choice(examples, size=this_sz, replace=False)
					token_probas = None
					if t_name is not 'BERT': 
						# (not pure-transform) Take a relative approach. We want to learn a delta ontop of bertOp
						# (pure-transform)     Take an absolute approach. We want pure-samples and not based on bertOp
						token_probas = searchOpts.get_bert_relative(1, t_name, tform)

					args = tform_configs, this_egs, token_probas, rep_tform, rep_id, searchOpts, ds
					batch, config_dict = self.process_examples(*args)
					aggregated_data.append((batch, config_dict))
		return aggregated_data

	def process_examples(
			self, this_ds_configs, examples, token_probas, representation_tform, 
			rep_id, searchOpts, ds
	):
		_, stage_map = searchOpts.config.get_stage_w_name(1)
		(inputs, labels, masks_for_tformed), supervised_labels = self.collate(examples, token_probas)
		pad_mask = 1.0 - (inputs.eq(self.dataOpts.tokenizer.pad_token_id)).float()
		rep_mask = representation_tform.get_rep_tform(inputs.shape, pad_mask, rep_id)
		batch = {'input': inputs, 'output': None, 'rep_mask': rep_mask}
		config_dict = {}
		for config_ in this_ds_configs:
			token_tform_name = stage_map[config_[1]]
			task_output = masks_for_tformed[token_tform_name][1]
			# We are assuming that the out-type will be the same as the ds-type in the supervised setting
			is_supervised = searchOpts.config.get_name(3, config_[-1]) == searchOpts.config.get_name(0, config_[0])
			if is_supervised:
				assert ds.is_supervised, 'We can only have this setting for supervised data'
				task_output = supervised_labels
			out_type = searchOpts.config.get_name(3, config_[-1])
			dict_ = self.apply_out_tform(out_type, ds, inputs, task_output, examples, is_supervised=is_supervised)
			config_dict[config_] = dict_['output']
		return batch, config_dict

	def get_seed_emb(self):
		if self.prim_task_id == 'chemprot':
			# seeds = ['chemistry', 'inhibitor', 'cells', 'acid', 'receptor', 'protein', 'kinase', 'phosphorylation', 'antagonist', 'enzyme', 'activity', 'mrna', 'synthase', 'tyrosine', 'agonist','synthesis', 'binding', 'apoptosis', 'acetylcholinesterase', 'cytochrome']
			# input_chem = self.dataOpts.tokenizer_bert.batch_encode_plus(seeds[:self.num_seed], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True) # (expert)
			# input_chem = self.dataOpts.tokenizer_bert.batch_encode_plus(['Understanding pharmacological agents and mechanisms', 'Identifying and classifying pharmacological agents','Interpreting scientific and technical language','Understanding results of scientific experiments','Classifying actions of pharmacological agents'], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True) # (chatGPT)
			# input_chem = self.dataOpts.tokenizer_bert.batch_encode_plus(['activity', 'inhibitor', 'expression', 'induced', 'cells', 'acid', 'inhibition', 'receptor', 'kinase', 'protein', 'inhibited', 'treatment', 'inhibitors', 'increased', 'also'], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True) # (15 words) tf-idf
			# input_chem = self.dataOpts.tokenizer_bert.batch_encode_plus(['activity', 'inhibitor', 'expression', 'induced', 'cells', 'acid', 'inhibition', 'receptor', 'kinase', 'protein', 'inhibited', 'treatment', 'inhibitors', 'increased'], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True) # (14 words) tf-idf-2
			# input_chem = self.dataOpts.tokenizer_bert.batch_encode_plus(['activity', 'inhibitor', 'expression', 'induced', 'cells', 'acid', 'inhibition', 'receptor', 'kinase', 'protein'], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True) # (10 words) tf-idf-3
			# input_chem = self.dataOpts.tokenizer_bert.batch_encode_plus(['activity', 'inhibitor', 'expression', 'induced', 'cells', 'acid', 'inhibition', 'receptor', 'kinase', 'protein', 'inhibited', 'treatment', 'inhibitors', 'also', 'increased', 'human','dependent', 'alpha', 'cox', 'activation'], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True) # (20 words) tf-idf-4 (best)
			# input_chem = self.dataOpts.tokenizer_bert.batch_encode_plus(['activity', 'inhibitor', 'expression', 'induced', 'cells', 'acid', 'inhibition', 'receptor', 'kinase', 'protein', 'inhibited', 'treatment', 'inhibitors', 'also', 'increased', 'human', 'dependent', 'alpha', 'cox', 'activation', 'effects', 'selective', 'ca', 'levels', 'il', 'factor', 'antagonist', 'microm', 'beta','effect'], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True) # (30 words) tf-idf-5
			# seeds = ['activity', 'inhibitor', 'expression', 'induced', 'cells', 'acid', 'inhibition', 'receptor', 'kinase', 'protein', 'inhibited', 'treatment', 'inhibitors', 'also', 'increased', 'human','dependent', 'alpha', 'cox', 'activation']
			# input_chem = self.dataOpts.tokenizer_bert.batch_encode_plus(seeds[:self.num_seed], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True) # (table_spreedsheet)
			seeds = ['inos','adrenoceptor','inhibit','tcdd','akt','agonist', 'vegf', 'caspase', 'inhibiting', 'ec','cyclin', 'microm', 'antagonist', 'potency', 'antagonists', 'inhibitory', 'potent', 'inhibits', 'tyrosine', 'synthase']
			input_chem = self.dataOpts.tokenizer_bert.batch_encode_plus(seeds[:self.num_seed], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)
            
            
		if self.prim_task_id == 'citation_intent':
			# seeds = ['language', 'text', 'model', 'information', 'grammar', 'lexical', 'translation', 'learning', 'word', 'corpus', 'semantic', 'syntactic', 'features', 'structure', 'data', 'training', 'rules', 'analysis', 'parser', 'statistical']
			# input_chem = self.dataOpts.tokenizer_bert.batch_encode_plus(seeds[:self.num_seed], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True) # (expert)
			# input_chem = self.dataOpts.tokenizer_bert.batch_encode_plus(['Natural language processing','Error rate reduction','Embodied communication agents','Parsing algorithms','Diagnostic systems'], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True) # (chatGPT)
			# input_chem = self.dataOpts.tokenizer_bert.batch_encode_plus(['based', 'work', 'used', 'system','using', 'use', 'model','information', 'word', 'language', 'also', 'grammar', 'approach', 'translation', 'corpus'], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)
			# seeds = ['based', 'work', 'used', 'system', 'using', 'use', 'model', 'information', 'word', 'language', 'also', 'grammar', 'approach','translation', 'corpus', 'collins', 'lexical', 'semantic', 'text','learning']
			# input_chem = self.dataOpts.tokenizer_bert.batch_encode_plus(seeds[:self.num_seed], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)
			seeds = ['charniak', 'coreference', 'treebank', 'parsers', 'grammars', 'parse', 'syntactic', 'linguistic', 'lexical', 'roth', 'och', 'parser', 'parsing', 'extraction', 'chen', 'sentences', 'phrase', 'semantic', 'sentence', 'dialogue']
			input_chem = self.dataOpts.tokenizer_bert.batch_encode_plus(seeds[:self.num_seed], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)
            
		output_chem = self.dataOpts.model_bert(**input_chem)
		return output_chem



	def get_similarity_entity(self, sent):
		output = ner_prediction(corpus=sent, compute='gpu')
		try:
			sim_word = output['value'].tolist()
		except:
			return []
		# K = int(0.15*len(sim_word))
		# sim_word = random.sample(sim_word, K)
		return sim_word
    

    
	def get_masked_indices(self, sent, tokenized_sent):
		tsim_word = set(self.get_similarity_entity(sent))
		tsim_word = filter(lambda x: [x for i in tsim_word if x in i and x != i] == [], tsim_word)
		sim_word = []
		sindex = {}

		for x in tsim_word:
			if x in sent:
				sim_word.append(x)

		for x in sim_word:
			sindex[x] = sent.find(x)
			sent = sent.replace(x, "%%%%%", 1)
		temp_words = [x for x in sent.split(' ')]
		a = sorted(sindex.items(), key=lambda x: x[1])
		slist = [x[0] for x in a]
		words = []
		idx = 0
		for x in temp_words:
			if "%%%%%" in x:
				try:
					words.append(slist[idx])
				except:
					print(temp_words, sim_word, slist, sent)
					raise
				idx += 1
			else:
				words.append(x)
  
		idx = 1
            
		enc = self.dataOpts.tokenizer.batch_encode_plus(words, add_special_tokens=False)
		desired_output = []
		masked_tok_indices = []
		masked_tok_dict = {}
		total_len = 0
		for w, token in zip(words, enc.input_ids):
			tokenoutput = []
			tokenoutput.extend(range(idx, idx+len(token)))
			idx += len(token)
			total_len += len(token)
			if w in sim_word:
				masked_tok_dict[w] = tokenoutput
		proba, total_sim_word = [], len(sim_word)    
		for idx, w in enumerate(sim_word):
			try:
				masked_tok_indices.extend(masked_tok_dict[w])
			except:
				print(sim_word, w)
				print(sent) 
				print(words) 
				print(masked_tok_dict)
				raise 
                
			proba.extend([total_sim_word-idx] * len(masked_tok_dict[w]))

		if len(masked_tok_indices) == 0:
			masked_tok_indices = random.sample(range(total_len), int(total_len*0.15))
		else:
			masked_tok_indices = masked_tok_indices[:int(total_len*0.15)]
		# proba = np.array(proba)/(1.0* np.array(proba).sum())
		masked_tok_indices = np.array(masked_tok_indices)#[np.random.choice(len(masked_tok_indices), size=int(total_len*self.proba), replace=False, p=proba)]
		mask = torch.full(tokenized_sent.shape, False)
		try:
			mask[masked_tok_indices] = True
		except:
			import pdb; pdb.set_trace()
		return mask
    
	def get_masked_indices_alpha(self, sent, tokenized_sent, alpha):
		sim_word = self.get_similarity_entity(sent)
		idx = 1
		words = [x for x in sent.split(' ')]
		enc = self.dataOpts.tokenizer.batch_encode_plus(words, add_special_tokens=False)
		desired_output = []
		masked_tok_indices = []
		masked_tok_dict = {}
		total_len = 0
		for w, token in zip(words, enc.input_ids):
			tokenoutput = []
			tokenoutput.extend(range(idx, idx+len(token)))
			idx += len(token)
			total_len += len(token)
			if w in sim_word:
				masked_tok_dict[w] = tokenoutput
		proba, total_sim_word = [], len(sim_word)    
		for idx, w in enumerate(sim_word):
			masked_tok_indices.extend(masked_tok_dict[w])
			proba.extend([total_sim_word-idx] * len(masked_tok_dict[w]))
		# maksed_tok_indices = maksed_tok_indices[:int(total_len*0.15)]
		proba = np.array(proba)/(1.0* np.array(proba).sum())
		proba = (proba*alpha) + (np.repeat(1.0/(len(proba)*1.0), [len(proba)])*(1.0-alpha))
		masked_tok_indices = np.array(masked_tok_indices)[np.random.choice(len(masked_tok_indices), size=int(total_len*self.proba), replace=False, p=proba)]
		mask = torch.full(tokenized_sent.shape, False)
		try:
			mask[masked_tok_indices] = True
		except:
			import pdb; pdb.set_trace()
		return mask


	def collate(self, examples, token_probas):
		# import pdb; pdb.set_trace()        
		all_egs_text = [x['sample'] for x in examples]
		out = self.dataOpts.tokenizer.batch_encode_plus(
								all_egs_text, add_special_tokens=True, truncation=True,
								max_length=self.block_size, return_special_tokens_mask=True
				)
		all_egs = [torch.tensor(x) for x in out["input_ids"]]
		labels = None
		if 'label' in examples[0]:
			labels = torch.tensor([x['label'] for x in examples])
		inputs = pad_sequence(all_egs, self.dataOpts.tokenizer.pad_token_id)
		if self.has_alpha:
			maksed_tok_indices = torch.stack([self.get_masked_indices_alpha(x['sample'], inputs[idx], self.alpha) for idx, x in enumerate(examples)])
		else:
			maksed_tok_indices = torch.stack([self.get_masked_indices(x['sample'], inputs[idx]) for idx, x in enumerate(examples)])
		# maksed_tok_indices = self.get_masked_indices(all_egs_text, )
		return self.apply_in_tform(maksed_tok_indices, inputs, token_probas), labels

# Todo  [ldery] - run some tests to make sure code here is working
def run_tests():
	import argparse
	from config import Config
	from transformers import (
		AutoTokenizer,
		PreTrainedTokenizer,
	)
	parser = argparse.ArgumentParser()
	add_data_args(parser)
	args = parser.parse_args()
	args.task_data = '/home/ldery/projects/AutoAuxiliaryLoss/test_datasets/citation_intent.dev.txt'
	args.in_domain_data = '/home/ldery/projects/AutoAuxiliaryLoss/test_datasets/test.json'
	print('Getting Config')
	autoloss_config = Config('full')
	print('Getting Tokenizer')
	tokenizer = AutoTokenizer.from_pretrained('roberta-base')
	# Taking that the config stage 0 is the input stage
	print('Getting aux_dataOptions')
	try:
		print('Some config entries not present. Should throw error')
		aux_dataOptions = DataOptions(args, tokenizer, autoloss_config.get_stage(0), autoloss_config.get_stage(-1))
		msg = 'Failed.'
		print("test datasets match config: {}".format(msg))
		exit()
	except:
		msg = 'Passed.'
	print("test datasets match config: {}".format(msg))
	args.out_domain_data = '/home/ldery/projects/AutoAuxiliaryLoss/test_datasets/sciie.dev.txt'
	args.neural_lm_data = '/home/ldery/projects/AutoAuxiliaryLoss/test_datasets/chemprot.dev.txt'
	
	try:
		print('Able to load full options and process all data')
		aux_dataOptions = DataOptions(args, tokenizer, autoloss_config.get_stage(0), autoloss_config.get_stage(-1))
		msg = 'Passed.'
	except:
		msg = 'Failed.'
		print("test can load data option: {}".format(msg))
		exit()
	print("test can load data option: {}".format(msg))

	try:
		print('Checking doc ids')
		ds = aux_dataOptions.get_dataset(1)
		assert ds.getdocid(0) == 'CHEMPROT', 'Checking correct doc_id for 0'
		assert ds.getdocid(2442) == 'CITATION', 'Checking correct doc_id for 2542'
		assert ds.getdocid(2542) == 'SCIIE', 'Checking correct doc_id for 2542'
		assert ds.getdocid(3109) == 'CITATION.1', 'Checking correct doc_id for 3109'
		msg = 'Passed.'
	except:
		msg = 'Failed.'
		print("test correct doc ids: {}".format(msg))
		exit()
	print("test correct doc ids: {}".format(msg))
	print('==='*5, 'DataOptions Tests Passed. Moving on to DataTransformAndItr', '==='*5)
	setattr(args, 'per_gpu_train_batch_size', 32)
	setattr(args, 'n_gpu', 1)
	setattr(args, 'mlm_probability', 0.15)
	setattr(args, 'block_size', 512)

	try:
		dtform_and_itr = DataTransformAndItr(args, aux_dataOptions, autoloss_config.get_stage(1), autoloss_config.get_stage(-1))
		msg = 'Passed.'
	except:
		msg = 'Failed.'
		print("test Init DataTransformAndItr: {}".format(msg))
		exit()
	print("test Init DataTransformAndItr: {}".format(msg))
	# Main approach to testing this was checking if the outputs look reasonable
	print('Input Should be same as output for chosen indices')
	loss_config = (1, 0, 0, 0)
	itr_ = dtform_and_itr.get_iterator(loss_config)
	for k in itr_:
		inp, out = k['input'], k['output']
	print('Visually inspected and works')
	
	print('Input Should be different token as output for chosen indices')
	loss_config = (1, 1, 0, 0)
	itr_ = dtform_and_itr.get_iterator(loss_config)
	for k in itr_:
		inp, out = k['input'], k['output']
	print('Visually inspected and works')
	
	print('Input Should be have mask token as output for chosen indices')
	loss_config = (1, 2, 0, 0)
	itr_ = dtform_and_itr.get_iterator(loss_config)
	for k in itr_:
		inp, out = k['input'], k['output']
	print('Visually inspected and works')
	print('Input Should be have mask token as output for chosen indices and tfids for output')
	loss_config = (1, 2, 0, 1)
	itr_ = dtform_and_itr.get_iterator(loss_config)
	for k in itr_:
		inp, out = k['input'], k['output']
	print('Visually inspected and works')
	pdb.set_trace()

	print('Input Should be have mask token as output for chosen indices and capitalization for output')
	loss_config = (1, 2, 0, 3)
	itr_ = dtform_and_itr.get_iterator(loss_config)
	for k in itr_:
		inp, out = k['input'], k['output']
	print('Visually inspected and works')
	pdb.set_trace()

	print('Input Should be have mask token as output for chosen indices and term-frequencies for output')
	loss_config = (1, 2, 0, 2)
	itr_ = dtform_and_itr.get_iterator(loss_config)
	for k in itr_:
		inp, out = k['input'], k['output']
	print('Visually inspected and works')


if __name__ == '__main__':
	run_tests()