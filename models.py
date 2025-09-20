# models.py

import torch
import torch.nn as nn
from torch import optim
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class FFNN(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.ln = nn.Linear(300, 300)
        self.relu = nn.ReLU()
        self.ln2 = nn.Linear(300, 1)
        self.logit = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, 1)
        x = self.ln(x)
        x = self.relu(x)
        x = self.ln2(x)
        x = self.logit(x)
        return x


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """

    def __init__(self, ffnn: FFNN, word_embeddings: WordEmbeddings, max_length):
        self.ffnn = ffnn
        self.word_embeddings = word_embeddings
        self.max_length = max_length

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        # Will need to step through this and tidy it up - nesting and indexes aren't quite right but should be close
        train_exs_word_indices = [self.word_embeddings.word_indexer.index_of(word) for word in ex_words]
        fixed_train_exs_word_indices = [ex if ex != -1 else 1 for ex in train_exs_word_indices]
        padded_train_exs_word_indices = fixed_train_exs_word_indices + [0]*(self.max_length - len(fixed_train_exs_word_indices))
        batch_data = torch.tensor([padded_train_exs_word_indices])

        model = self.ffnn.forward(batch_data)

        return model > 0.5

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """

    #random.seed(12)
    #np.random.seed(12)
    #torch.manual_seed(12)

    # Convert words to word indexes
    train_exs_word_indices = [[word_embeddings.word_indexer.index_of(word) for word in ex.words] for ex in train_exs]

    # Replace any -1's with 1's for unknown
    fixed_train_exs_word_indices = [[index if index != -1 else 1 for index in ex] for ex in train_exs_word_indices]

    # Retrieve the length of the longest sentence
    max_length = len(max(fixed_train_exs_word_indices, key = len))

    # Pad with 0's
    padded_train_exs_word_indices = [sublist + [0]*(max_length - len(sublist)) for sublist in fixed_train_exs_word_indices]

    # Create architecture
    ffnn: FFNN = FFNN(word_embeddings.get_initialized_embedding_layer(False, 0))
    optimizer = optim.Adam(ffnn.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    # Loop over epochs
    for epoch in range(args.num_epochs):

        total_loss = 0.0

        # Define training example indices
        epoch_iteration_indices = [i for i in range(len(train_exs))]
        random.shuffle(epoch_iteration_indices)

        # Define batches
        epoch_batch_indices = [epoch_iteration_indices[i:i+args.batch_size] for i in range(0, len(epoch_iteration_indices), args.batch_size)]
        #epoch_batch_indices = [[epoch_iteration_indices[i]] for i in epoch_iteration_indices]

        batched_data = [[torch.tensor([padded_train_exs_word_indices[index] for index in batch]),
                         torch.tensor([train_exs[index].label for index in batch])] for batch in epoch_batch_indices]
        # torched_data = torch.from_numpy(batched_data)

        for batch_data, batch_labels in batched_data:
            ffnn.zero_grad()
            log_probs = ffnn.forward(batch_data)
            # log_probs = torch.squeeze(log_probs)
            # log_probs = log_probs.unsqueeze(0)
            # loss = loss_fn(log_probs, batch_labels.to(torch.float32))
            loss = loss_fn(log_probs, batch_labels.unsqueeze(1).to(torch.float32))
            total_loss += loss
            loss.backward()
            optimizer.step()

        print("Total loss on epoch %i: %f" % (epoch, total_loss))


    return NeuralSentimentClassifier(ffnn, word_embeddings, max_length)
