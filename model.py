import torch.nn as nn
import torch

class RNN(nn.Module):
	
	def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
		"""
		Initialize the PyTorch RNN Module
		:param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
		:param output_size: The number of output dimensions of the neural network
		:param embedding_dim: The size of embeddings, should you choose to use them        
		:param hidden_dim: The size of the hidden layer outputs
		:param dropout: dropout to add in between LSTM/GRU layers
		"""
		super(RNN, self).__init__()        
		# set class variables
		self.vocab_size = vocab_size
		self.output_size = output_size
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		self.dropout = dropout
		# define model layers
		self.embed = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
		self.fc = nn.Linear(hidden_dim, output_size)
	
	
	def forward(self, nn_input, hidden):
		"""
		Forward propagation of the neural network
		:param nn_input: The input to the neural network
		:param hidden: The hidden state        
		Return: Two Tensors, the output of the neural network and the latest hidden state
		"""
		# return one batch of output word scores and the hidden state
		output = self.embed(nn_input)
		output, hidden = self.lstm(output, hidden)
		output = output.contiguous().view(-1, self.hidden_dim)
		output = self.fc(output)
		output = output.view(nn_input.shape[0], -1, self.output_size) 
		# This is the last prediction of a sequence
		return output[:,-1], hidden
	
	
	def init_hidden(self, batch_size):
		'''
		Initialize the hidden state of an LSTM/GRU
		:param batch_size: The batch_size of the hidden state
		Return: hidden state of dims (n_layers, batch_size, hidden_dim)
		'''
		# Implement function
		weight = next(self.parameters()).data
		
		hidden_state = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
		cell_state = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

		if (torch.cuda.is_available()):
			hidden = (hidden_state.cuda(), cell_state.cuda())
		else:
			hidden = (hidden_state, cell_state)
		# initialize hidden state with zero weights, and move to GPU if available
		return hidden