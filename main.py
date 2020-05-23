from model import RNN
from utils import *
import torch
import torch.nn as nn

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sequence_length', default=50, type=int)
parser.add_argument('--batch_size', default=250, type=int)
parser.add_argument('--num_epochs', default=12, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--embedding_dim', default=300, type=int)
parser.add_argument('--hidden_dim', default=512, type=int)
parser.add_argument('--n_layers', default=3, type=int)
parser.add_argument('--show_every_n_batches', default=500, type=int)
parser.add_argument('--data_dir', default='./data/Seinfeld_Scripts.txt', type=str)
opt = parser.parse_args()

# Check if GPU available
train_on_gpu = torch.cuda.is_available()

preprocess_and_save_data(opt.data_dir)
int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()
vocab_size = len(list(set(int_text)))
output_size = len(list(set(int_text)))
train_loader = batch_data(int_text, opt.sequence_length, opt.batch_size)


def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
	"""
	Implement forward and backward propagation on the model
	"""
	
	if (torch.cuda.is_available()):
		rnn.cuda()
		inp, target = inp.cuda(), target.cuda()

	hidden = tuple([each.data for each in hidden])
	optimizer.zero_grad()
	output, hidden = rnn.forward(inp, hidden)
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()
	return loss.item(), hidden


def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
	"""
	Train the model and return the trained model
	"""
	batch_losses = []
	
	rnn.train()

	print("Training for %d epoch(s)..." % n_epochs)
	for epoch_i in range(1, n_epochs + 1):
		hidden = rnn.init_hidden(batch_size)
		
		for batch_i, (inputs, labels) in enumerate(train_loader, 1):
			
			n_batches = len(train_loader.dataset)//batch_size
			if(batch_i > n_batches):
				break
			
			loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
			batch_losses.append(loss)

			if batch_i % show_every_n_batches == 0:
				print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
					epoch_i, n_epochs, np.average(batch_losses)))
				batch_losses = []

	return rnn


rnn = RNN(vocab_size, output_size, opt.embedding_dim, opt.hidden_dim, opt.n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

optimizer = torch.optim.Adam(rnn.parameters(), lr=opt.learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, opt.batch_size, optimizer, criterion, opt.num_epochs, opt.show_every_n_batches)

# saving the trained model
save_model('./save/trained_rnn', trained_rnn)
print('Model Trained and Saved')



