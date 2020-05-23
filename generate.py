import torch.nn.functional as F 
from utils import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gen_length', default=300, type=int)
parser.add_argument('--prime_word', default='love', type=str)
parser.add_argument('--top_k', default=5, type=int)
parser.add_argument('--model_dir', default='./save/trained_rnn', type=str)
parser.add_argument('--sequence_length', default=50, type=int)

opt = parser.parse_args()

# Load trained model
trained_rnn = load_model(opt.model_dir)
# Load pre-processed text
int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()
train_on_gpu = torch.cuda.is_available()

def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=opt.gen_length):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    Return: The generated text
    """
    rnn.eval()
    
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, opt.sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    for _ in range(predict_len):
        if train_on_gpu:
            rnn.cuda()
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = opt.top_k
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)     
        
        if(train_on_gpu):
            current_seq = current_seq.cpu() # move to cpu
        # the generated word becomes the next "current sequence" and the cycle can continue
        if train_on_gpu:
            current_seq = current_seq.cpu()
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    # return all the sentences
    return gen_sentences


pad_word = SPECIAL_WORDS['PADDING']
generated_script = generate(trained_rnn, vocab_to_int[opt.prime_word], int_to_vocab, token_dict, vocab_to_int[pad_word], opt.gen_length)
print(generated_script)