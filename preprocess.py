import random
import torch

def filter_sentence(sentence):
  """
  Filter sentence from unnecessary symbols. 

  Parameters
  ----------
  sentence: str
      Sentence to filter.
  
  Returns
  -------
  sentence: str
      Filtered sentence.
  """
  return ''.join(list(filter(list(char2idx).__contains__, sentence)))


def sentence2nums(sentence):
  """
  Map sentence to vector of nums and add EOS token to end.

  Parameters
  ----------
  sentence: str
      Sentence to map to vector of numbers.
  
  Returns
  -------
  vector: list of ints
      Vector of numbers mapped from sentence.
  """
  return [char2idx[x] for x in sentence] + [EOS_token]


def nums2sentence(nums):
  """
  Map vector of numbers to sentence.

  Parameters
  ----------
  nums: list of ints
      Vector of numbers to map to sentence.
  
  Returns
  -------
  sentence: str
      Sentence mapped from vector of numbers.
  """
  return ''.join([idx2char[x] for x in nums])


def add_mistakes(sentence):
  """
  Add mistakes to sentence.

  Parameters
  ----------
  sentence: str
      A sentence to add mistakes to.
  
  Returns
  -------
  sentence: str
      A sentence with mistakes added to it.
  """
  sentence = list(sentence)
  # delete random symbol
  if random.random() > 0.5:
    idx = random.randint(0, len(sentence) - 1)
    sentence = sentence[:idx] + sentence[idx + 1:]
  # replace random symbol with random symbol
  if random.random() > 0.5:
    idx = random.randint(0, len(sentence) - 1)
    idx2 = random.randint(0, len(alphabet) - 1)
    sentence[idx] = alphabet[idx2]
  # replace two random symbols
  if random.random() > 0.5:
    idx = random.randint(0, len(sentence) - 1)
    idx2 = random.randint(0, len(sentence) - 1)
    sentence[idx], sentence[idx2] = sentence[idx2], sentence[idx]
  # delete random whitespace
  if random.random() > 0.5:
    idxs = []
    for i, s in enumerate(sentence):
      if s == ' ':
        idxs.append(i)
    idx = random.choice(idxs)
    sentence = sentence[:idx] + sentence[idx + 1:]
  return ''.join(sentence)


def form_batches(lines, batch_size, num_of_words):
  """Form batches and targets from string.
  Every batch has `batch_size` sentences.
  Every sentence has about `num_of_words` words.

  Parameters
  ----------
  lines: str
      String that contains text
  batch_size: int
      Size of batch
  num_of_words: int
      Number of words in every sentence
  
  Returns
  -------
  batches: list of lists of torch.tensors
      Created batches
  batches_lengths: list of lists of lists of ints
      Lengths of created sentences
  targets: list of lists of torch.tensors
      Created targets (same to batches but without mistakes)
  targets_lengths: list of lists of lists of ints
      Lengths of target sentences
  """
  lines = filter_sentence(lines.lower())
  # create batches
  batches_lengths = []
  targets_lengths = []
  target_batch_lengths = []
  batch_lengths = []
  targets = []
  target_batch = []
  batches = []
  cnt = 0
  one_sentence = []
  batch = []
  for symbol in lines:
    if symbol == ' ':
      cnt += 1
    one_sentence.append(symbol)
    if cnt == num_of_words + 1:
      # one sentence is full
      cnt = 0
      one_sentence = ''.join(one_sentence)
      target_batch.append(torch.tensor(sentence2nums(one_sentence)))
      target_batch_lengths.append(len(one_sentence))
      one_sentence = add_mistakes(one_sentence)
      one_sentence = add_mistakes(one_sentence)
      batch.append(torch.tensor(sentence2nums(one_sentence)))
      batch_lengths.append(len(one_sentence))
      one_sentence = []
      if len(batch) == batch_size:
        # one batch is full
        batches.append(batch)
        batch = []
        targets.append(target_batch)
        target_batch = []
        batches_lengths.append(batch_lengths)
        targets_lengths.append(target_batch_lengths)
        batch_lengths = []
        target_batch_lengths = []
  return batches, batches_lengths, targets, targets_lengths


def get_str(lenta):
  """Get string of 1000 articles from Lenta.ru

  Parameters
  ----------
  lenta: generator
      Generator of Lenta.ru articles
  
  Returns
  -------
  line: str
      String of 1000 articles
  """
  articles = []
  for _ in range(1000):
    articles.append(next(lenta).text)
  line = ' '.join(articles)
  return line
