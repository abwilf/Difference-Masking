from copy import deepcopy

BERT = {
	'input-space' : ['Task'],
	'input-tform-space': ['BERT'],
	'rep-tform-space' : ['None'],
	'out-token-space' : ['DENOISE'],
	'out-sent-space': [],
}
BERT['out-space'] = [*BERT['out-token-space'], *BERT['out-sent-space']]


TAPT = {
	'input-space' : ['Task'],
	'input-tform-space': ['BERT'],
	'rep-tform-space' : ['None'],
	'out-token-space' : ['DENOISE'],
	'out-sent-space': [],
}
TAPT['out-space'] = [*TAPT['out-token-space'], *TAPT['out-sent-space']]

GPT = {
	'input-space' : ['Task'],
	'input-tform-space': ['None'],
	'rep-tform-space' : ['Left-To-Right'],
	'out-token-space' : ['DENOISE'],
	'out-sent-space': [],
}
GPT['out-space'] = [*GPT['out-token-space'], *GPT['out-sent-space']]

XLNET = {
	'input-space' : ['Task'],
	'input-tform-space': ['None'],
	'rep-tform-space' : ['Random-Factorized'],
	'out-token-space' : ['DENOISE'],
	'out-sent-space': [],
}
XLNET['out-space'] = [*XLNET['out-token-space'], *XLNET['out-sent-space']]


JOINTBASIC = {
	'input-space' : ['Task'],
	'input-tform-space': ['None', 'BERT'],
	'rep-tform-space' : ['None', 'Random-Factorized', 'Left-To-Right'],
	'out-token-space' : ['DENOISE'],
	'out-sent-space': [],
	'illegal_pairs': [
						('None', 'None'),
						('BERT', 'Random-Factorized'),
						('BERT', 'Left-To-Right'),
					 ]
}
JOINTBASIC['out-space'] = [*JOINTBASIC['out-token-space'], *JOINTBASIC['out-sent-space']]



VBASIC = {
	'input-space' : ['Task', 'In-Domain'],
	'input-tform-space': ['None', 'Replace', 'Mask'],
# 	'rep-tform-space' : ['None', 'Random-Factorized'],
	'rep-tform-space' : ['None', 'Left-To-Right', 'Right-To-Left', 'Random-Factorized'],
	'out-token-space' : ['DENOISE'],
	'out-sent-space': [],
}
VBASIC['out-space'] = [*VBASIC['out-token-space'], *VBASIC['out-sent-space']]


VBASIC1 = {
	'input-space' : ['Task'],
	'input-tform-space': ['None', 'Replace', 'Mask', 'BERT'],
# 	'rep-tform-space' : ['None', 'Random-Factorized'],
	'rep-tform-space' : ['None', 'Left-To-Right', 'Right-To-Left', 'Random-Factorized'],
	'out-token-space' : ['DENOISE'],
	'out-sent-space': [],
}
VBASIC1['out-space'] = [*VBASIC1['out-token-space'], *VBASIC1['out-sent-space']]

SUPERVISED = {
	'input-space' : ['MNLI'],
	'input-tform-space': ['None'],
	'rep-tform-space' : ['None', 'Random-Factorized'],
	'out-token-space' : ['MNLI'],
	'out-sent-space': [],
}
SUPERVISED['out-space'] = [*SUPERVISED['out-token-space'], *SUPERVISED['out-sent-space']]



CITATION_ALL = {
	'input-space' : ['CITATION_INTENT', 'In-Domain'],
	'input-tform-space': ['None', 'Replace', 'Mask', 'BERT'],
	'rep-tform-space' : ['None', 'Left-To-Right', 'Right-To-Left', 'Random-Factorized'],
	'out-token-space' : ['DENOISE'],
	'out-sup-space':['CITATION_INTENT'],
	'out-sent-space': [],
}
CITATION_ALL['out-space'] = [*CITATION_ALL['out-sup-space'], *CITATION_ALL['out-token-space'], *CITATION_ALL['out-sent-space']]

CITATION_EXT = {
	'input-space' : ['In-Domain'],
	'input-tform-space': ['None', 'Replace', 'Mask', 'BERT'],
	'rep-tform-space' : ['None', 'Left-To-Right', 'Right-To-Left', 'Random-Factorized'],
	'out-token-space' : ['DENOISE'],
	'out-sup-space':[],
	'out-sent-space': [],
}
CITATION_EXT['out-space'] = [*CITATION_EXT['out-sup-space'], *CITATION_EXT['out-token-space'], *CITATION_EXT['out-sent-space']]



SCIIE_ALL = {
	'input-space' : ['SCIIE', 'In-Domain'],
	'input-tform-space': ['None', 'Replace', 'Mask', 'BERT'],
	'rep-tform-space' : ['None', 'Left-To-Right', 'Right-To-Left', 'Random-Factorized'],
	'out-token-space' : ['DENOISE'],
	'out-sup-space':['SCIIE'],
	'out-sent-space': [],
}
SCIIE_ALL['out-space'] = [*SCIIE_ALL['out-sup-space'], *SCIIE_ALL['out-token-space'], *SCIIE_ALL['out-sent-space']]


SEMEVAL_ALL = {
	'input-space' : ['SemEval2016Task6', 'MNLI'],
	'input-tform-space': ['None', 'Replace', 'Mask', 'BERT'],
	'rep-tform-space' : ['None', 'Left-To-Right', 'Right-To-Left', 'Random-Factorized'],
	'out-token-space' : ['DENOISE'],
	'out-sup-space':['SemEval2016Task6', 'MNLI'],
	'out-sent-space': [],
}
SEMEVAL_ALL['out-space'] = [*SEMEVAL_ALL['out-sup-space'], *SEMEVAL_ALL['out-token-space'], *SEMEVAL_ALL['out-sent-space']]




CHEMPROT_ALL = {
	'input-space' : ['CHEMPROT', 'In-Domain', 'MNLI'],
	'input-tform-space': ['None', 'Replace', 'Mask'],
	'rep-tform-space' : ['None', 'Left-To-Right', 'Right-To-Left', 'Random-Factorized'],
	'out-token-space' : ['DENOISE'],
	'out-sup-space':['CHEMPROT', 'MNLI'],
	'out-sent-space': [],
}
CHEMPROT_ALL['out-space'] = [*CHEMPROT_ALL['out-sup-space'], *CHEMPROT_ALL['out-token-space'], *CHEMPROT_ALL['out-sent-space']]




CITATION_SUPERVISED = deepcopy(VBASIC)
CITATION_SUPERVISED['input-space'] = ['CITATION_INTENT']
CITATION_SUPERVISED['out-sup-space'] = ['CITATION_INTENT']
CITATION_SUPERVISED['input-tform-space'].extend(['BERT'])
CITATION_SUPERVISED['out-space'] = [*CITATION_SUPERVISED['out-sup-space'], *CITATION_SUPERVISED['out-token-space'], *CITATION_SUPERVISED['out-sent-space']]

SCIIE_SUPERVISED = deepcopy(VBASIC)
SCIIE_SUPERVISED['input-space'] = ['SCIIE']
SCIIE_SUPERVISED['input-tform-space'].extend(['BERT'])
SCIIE_SUPERVISED['out-sup-space'] = ['SCIIE']
SCIIE_SUPERVISED['out-space'] = [*SCIIE_SUPERVISED['out-sup-space'], *SCIIE_SUPERVISED['out-token-space'], *SCIIE_SUPERVISED['out-sent-space']]

CHEMPROT_SUPERVISED = deepcopy(VBASIC)
CHEMPROT_SUPERVISED['input-space'] = ['CHEMPROT']
CHEMPROT_SUPERVISED['input-tform-space'].extend(['BERT'])
CHEMPROT_SUPERVISED['out-sup-space'] = ['CHEMPROT']
CHEMPROT_SUPERVISED['out-space'] = [*CHEMPROT_SUPERVISED['out-sup-space'], *CHEMPROT_SUPERVISED['out-token-space'], *CHEMPROT_SUPERVISED['out-sent-space']]

HYPERPARTISAN_SUPERVISED = deepcopy(VBASIC)
HYPERPARTISAN_SUPERVISED['input-space'] = ['HYPERPARTISAN']
HYPERPARTISAN_SUPERVISED['input-tform-space'].extend(['BERT'])
HYPERPARTISAN_SUPERVISED['out-sup-space'] = ['HYPERPARTISAN']
HYPERPARTISAN_SUPERVISED['out-space'] = [*HYPERPARTISAN_SUPERVISED['out-sup-space'], *HYPERPARTISAN_SUPERVISED['out-token-space'], *HYPERPARTISAN_SUPERVISED['out-sent-space']]

SEMEVAL_SUPERVISED = deepcopy(VBASIC)
SEMEVAL_SUPERVISED['input-space'] = ['SemEval2016Task6']
SEMEVAL_SUPERVISED['input-tform-space'].extend(['BERT'])
SEMEVAL_SUPERVISED['out-sup-space'] = ['SemEval2016Task6']
SEMEVAL_SUPERVISED['out-space'] = [*SEMEVAL_SUPERVISED['out-sup-space'], *SEMEVAL_SUPERVISED['out-token-space'], *SEMEVAL_SUPERVISED['out-sent-space']]

PERSPECTRUM_SUPERVISED = deepcopy(VBASIC)
PERSPECTRUM_SUPERVISED['input-space'] = ['PERSPECTRUM']
PERSPECTRUM_SUPERVISED['out-sup-space'] = ['PERSPECTRUM']
PERSPECTRUM_SUPERVISED['out-space'] = [*PERSPECTRUM_SUPERVISED['out-sup-space'], *PERSPECTRUM_SUPERVISED['out-token-space'], *PERSPECTRUM_SUPERVISED['out-sent-space']]


BASIC = {
	'input-space' : ['Task'],
	'input-tform-space': ['None', 'Replace', 'Mask'],
	'rep-tform-space' : ['None'],
	'out-token-space' : ['DENOISE', 'TF'],
	'out-sent-space': [],
}
BASIC['out-space'] = [*BASIC['out-token-space'], *BASIC['out-sent-space']]


SBASIC = {
	'input-space' : ['In-Domain'],
	'input-tform-space': ['None', 'Replace', 'Mask'],
	'rep-tform-space' : ['None'],
	'out-token-space' : ['DENOISE'],
	'out-sent-space': ['FS'],
}
SBASIC['out-space'] = [*SBASIC['out-token-space'], *SBASIC['out-sent-space']]


W_ILLEGAL = deepcopy(BASIC)
W_ILLEGAL['rep-tform-space'].extend(['Left-To-Right'])
W_ILLEGAL['out-sent-space'].extend(['NSP', 'QT'])
W_ILLEGAL['out-space'] = [*W_ILLEGAL['out-token-space'], *W_ILLEGAL['out-sent-space']]


FULL = deepcopy(BASIC)
FULL['input-space'].extend(['Out-Domain', 'Neural-LM'])
FULL['input-tform-space'].extend(['Noisy-Embed'])
FULL['rep-tform-space'].extend(['Left-To-Right', 'Right-To-Left', 'Random-Factorized'])
FULL['out-token-space'].extend(['TF', 'CAP'])
FULL['out-sent-space'].extend(['QT', 'FS', 'ASP', 'SO', 'SCP', 'NSP'])
FULL['out-space'] = [*FULL['out-token-space'], *FULL['out-sent-space']]


ALL_TOKEN_OUTPUTS =  ['DENOISE', 'TFIDF', 'TF', 'CAP']
ALL_TOKEN_CLASSF = {
	'CAP': 2,
	'TFIDF': 1,
	'TF': 4,
}
ALL_SENT_CLASSF_OUTPUTS = {
	'NSP' : 2,
	'SO'  : 2,
	'ASP' : 3,
	'SCP' : 2
}
ALL_SENT_DOT_OUTPUTS = ['QT', 'FS']
# ALL_SUPERVISED_OUTPUTS = ['MNLI', 'CITATION_INTENT', 'SCIIE', 'CHEMPROT']

# NB - if the output space is supervised then the dataset name must be the same as the supervised set name
ALL_SUPERVISED_OUTPUTS = {
	'MNLI' : 3,
	'CITATION_INTENT'  : 6,
	'SCIIE': 7,
	'CHEMPROT': 13,
	'HYPERPARTISAN': 2,
	'SemEval2016Task6': 3,
	'PERSPECTRUM': 2
}

ALL_CONFIG_CHOICES = [
	'sbasic', 'basic', 'vbasic', 'vbasic1', 'with-illegal', 
	'bert', 'full', 'supervised', 'citation.supervised',
	'sciie.supervised', 'chemprot.supervised', 'tapt', 'semeval.all',
	'hyperpartisan.supervised', 'citation.all', 'sciie.all', 'SemEval2016Task6.supervised', 'chemprot.all',  'gpt', 'xlnet', 'jointbasic',
	'PERSPECTRUM.supervised', 'citation.ext'
]

def get_config(name):
	config = None
	if name == 'basic':
		config = BASIC
	elif name == 'sbasic':
		config = SBASIC
	elif name == 'with-illegal':
		config = W_ILLEGAL
	elif name == 'full':
		config = FULL
	elif name == 'vbasic':
		config = VBASIC
	elif name == 'vbasic1':
		config = VBASIC1
	elif name == 'supervised':
		config = SUPERVISED
	elif name == 'tapt':
		config = TAPT
	elif name == 'gpt':
		config = GPT
	elif name == 'xlnet':
		config = XLNET
	elif name == 'bert':
		config = BERT
	elif name == 'jointbasic':
		config = JOINTBASIC
	elif name == 'citation.supervised':
		config = CITATION_SUPERVISED
	elif name == 'sciie.supervised':
		config = SCIIE_SUPERVISED
	elif name == 'chemprot.supervised':
		config = CHEMPROT_SUPERVISED
	elif name == 'hyperpartisan.supervised':
		config = HYPERPARTISAN_SUPERVISED
	elif name == 'citation.all':
		config = CITATION_ALL
	elif name == 'citation.ext':
		config = CITATION_EXT
	elif name == 'sciie.all':
		config = SCIIE_ALL
	elif name == 'semeval.all':
		config = SEMEVAL_ALL
	elif name == 'chemprot.all':
		config = CHEMPROT_ALL
	elif name == 'SemEval2016Task6.supervised':
		config = SEMEVAL_SUPERVISED
	elif name == 'PERSPECTRUM.supervised':
		config = PERSPECTRUM_SUPERVISED
	assert config is not None, 'Wrong Config name given : {}'.format(name)
	return config

def get_config_options(name):
	config = get_config(name)
	return config['input-space'], config['input-tform-space'], \
			config['rep-tform-space'], config['out-space']

# Will need to expand this as more transforms are added
def get_illegal_sets(config):
	illegal_pairs = []
	if len(config['out-sent-space']) != 0:
		rep_w_issues = set(['Left-To-Right', 'Right-To-Left', 'Random-Factorized'])
		intersect = rep_w_issues.intersection(set(config['rep-tform-space']))
		if len(intersect) == 0:
			return []
		illegal_pairs = [(b, a) for a in config['out-sent-space'] for b in intersect]

	if ('out-sup-space' in config) and (len(config['out-sup-space']) != 0):
		rep_w_issues = set(['Left-To-Right', 'Right-To-Left'])
		intersect = rep_w_issues.intersection(set(config['rep-tform-space']))
		ill_rep = [(b, a) for a in config['out-sup-space'] for b in intersect]
		ill_ds_out = [(a, b) for a in config['input-space'] for b in config['out-sup-space'] if a != b]
		illegal_pairs.extend(ill_rep)
		illegal_pairs.extend(ill_ds_out)
	if 'illegal_pairs' in config:
		illegal_pairs.extend(config['illegal_pairs'])
	return illegal_pairs

def test_valid_config_name():
	try:
		get_config_options('None')
		msg = 'Failed.'
	except:
		msg = 'Passed.'
	print("test_valid_config_name : {}".format(msg))

def test_valid_config_keys():
	try:
		basic_keys = set(get_config('basic').keys())
		full_keys = set(get_config('full').keys())
		w_illegal_keys = set(get_config('with-illegal').keys())
		assert basic_keys == full_keys, 'Mismatch between basic and full'
		assert basic_keys == w_illegal_keys, 'Mismatch between basic and w_illegal'
		assert w_illegal_keys == full_keys, 'Mismatch between full and w_illegal'
		msg = 'Passed.'
	except:
		msg = 'Failed.'

	print("test_valid_config_keys : {}".format(msg))

def test_valid_illegal_set():
	try:
		w_illegal = get_config('with-illegal')
		illegal_pairs = get_illegal_sets(w_illegal)
		assert len(illegal_pairs) > 0, 'With Illegal Should have > 0 illegal pairings'
		
		full = get_config('full')
		illegal_pairs = get_illegal_sets(full)
		assert len(illegal_pairs) == 18, 'Invalid number of illegal options for full' # Hard coded
		msg = 'Passed.'
	except:
		msg = 'Failed.'

	print("test_valid_illegal_set : {}".format(msg))


def run_tests():
	test_valid_config_name()
	test_valid_config_keys()
	test_valid_illegal_set()

if __name__ == '__main__':
	run_tests()