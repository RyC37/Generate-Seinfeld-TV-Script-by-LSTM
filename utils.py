from collections import Counter
import os
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

SPECIAL_WORDS = {'PADDING': '<PAD>'}

def load_data(path):
	"""
	Load Dataset from File
	Return: a string contains the text data
	"""
	input_file = os.path.join(path)
	with open(input_file, "r") as f:
		data = f.read()

	return data

def create_lookup_tables(text):
	"""
	Create lookup tables for vocabulary
	Input: The text to be processed
	Return: A tuple of dicts (vocab_to_int, int_to_vocab)
	"""
	word_counts = Counter(text)
	sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
	int_to_vocab = dict(enumerate(sorted_vocab))
	vocab_to_int = {wd:i for i, wd in int_to_vocab.items()}
	
	return (vocab_to_int, int_to_vocab)

def token_lookup():
	"""
	Generate a dict to turn punctuation into a token.
	Return: Tokenized dictionary where the key is the punctuation and the value is the token
	"""
	punctuation_dict = {
		'.': '||Period||',
		',': '||Comma||',
		'"': '||Quotation_Mark||',
		';': '||Semicolon||',
		'!': '||Exclamation_Mark||',
		'?': '||Question_Mark||',
		'(': '||Left_Parentheses||',
		')': '||Right_Parentheses||',
		'-': '||Dash||',
		'\n': '||Return||'
	}
		
	return punctuation_dict

def preprocess_and_save_data(dataset_path):
	"""
	Preprocess text data, and save the data
	"""
	text = load_data(dataset_path)
	
	text = text[81:] # Ignore notice

	token_dict = token_lookup()
	for key, token in token_dict.items():
		text = text.replace(key, ' {} '.format(token))

	text = text.lower()
	text = text.split()

	vocab_to_int, int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
	int_text = [vocab_to_int[word] for word in text]
	pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def batch_data(words, sequence_length, batch_size):
	"""
	Batch the neural network data using DataLoader
	:param words: The word ids of the TV scripts
	:param sequence_length: The sequence length of each batch
	:param batch_size: The size of each batch; the number of sequences in a batch
	Return: DataLoader with batched data
	"""
	x = []
	y = []
	for i in range(len(words)-sequence_length):
		x.append(words[i:i+sequence_length])
		y.append(words[i+sequence_length])
	feature_tensor = torch.from_numpy(np.array(x))
	target_tensor = torch.from_numpy(np.array(y))
	dataset = TensorDataset(feature_tensor, target_tensor)
	return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_preprocess():
	"""
	Load the Preprocessed Training data
	"""
	return pickle.load(open('preprocess.p', mode='rb'))


def save_model(filename, decoder):
	save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
	torch.save(decoder, save_filename)


def load_model(filename):
	save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
	return torch.load(save_filename)