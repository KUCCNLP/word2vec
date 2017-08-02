#!/usr/bin/python3

# Import built-in modules
import collections, math, os, random, sys, zipfile

# Import external packages
from sklearn.manifold import TSNE
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print(sys.version)
print(sys.platform)

# Deactivate minor warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# File path to where the zipfile is located
FPATH_DATA = "../../../dev-data/kucc_nlpstudy/word2vec/text8.zip" 
FPATH_CKPT = "./logs/skipgram.ckpt"

# Define functions
def read_data(fpath):
	"""
	Read the data into a list of strings.
	Extract the first file enclosed in a zip file as a list of words.
	
	args:
		fpath	: str or pathlike-object
	return:
		data 		: list, list of
	"""
	with zipfile.ZipFile(fpath) as f: data = tf.compat.as_str(f.read(f.namelist()[0])).split()
	return data

def build_dataset(seq_word, num_words):
	"""
	Process raw inputs into a dataset.
	Build the dictionary and replace rare words with UNK token.

	args:
		seq_word 	: list, list of strings
		num_words 	: int, number of all vocabularies
	return:
		data 		: list, list of index
		dikt 		: dict, {word:index}
	"""
	seq_pair_word2freq = collections.Counter(seq_word).most_common(num_words - 1)
	dikt = {'UNK':0}
	for i, pair in enumerate(seq_pair_word2freq): dikt[pair[0]] = len(dikt)

	data = []
	for word in seq_word:
		if word in dikt: index = dikt[word]
		else: index = 0
		data.append(index)

	return data, dikt

def write_metadata(dikt, fname_metadata):
	str_fwrite = "index\tvalue\n"
	for k, v in dikt.items(): str_fwrite = "{0}{1}\t{2}\n".format(str_fwrite, v+1, k)
	with open(fname_metadata, "w") as fw: fw.write(str_fwrite)

def reverse_dikt(dikt):
	"""
	args: 
		dikt 		: dict,
	return:
		dikt_rvrs	: dict, keys and values are reversed
	"""
	dikt_rvrs = dict(zip(dikt.values(), dikt.keys()))
	return dikt_rvrs

def make_skip_index(skip_window):
	"""
	args:
		skip_window : int
	return:
		skip_index	: list, [-N, ..., -1, 1, ..., N]
	"""
	skip_index = [i for i in range(-skip_window, 0)] + [i for i in range(1, skip_window + 1)]
	return skip_index

def make_index_batch(batch_size, skip_window, index_data):
	"""
	args:
		skip_window : int
	return:
					: list, of shape(batch_size, 2)
	"""
	seq_index_batch = []
	for i in range(int(batch_size/skip_window/2)):
		for skip_index in make_skip_index(skip_window):
			seq_index_batch.append([(index_data + i)%len(data), (index_data + i + skip_index)%len(data)])
	return seq_index_batch

def make_batch(data, seq_index_batch):
	"""
	Specific computation to generate batch at each step

	args:
		data 		: list, list of integers(indeces to vocabulary)
		seq_index_batch	:
	return:
		batch 		: np.array, of shape=(batch_size)
		labels 		: np.array, of shape=(batch_size, 1)
	"""
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	for i, index_batch in enumerate(seq_index_batch):
		batch[i] = data[index_batch[0]]
		labels[i] = data[index_batch[1]]
	return batch, labels

def generate_batch(data, batch_size, skip_window, num_steps):
	"""
	Function to generate a training batch for each step

	args:
		data 		: list, list of integers(indeces to vocabulary)
		batch_size 	: int, size of the batch
		skip_window : int,
		num_steps 	: int, number of iterations to generate batch
	yield:
		batch 		: np.array, of shape=(batch_size)
		labels 		: np.array, of shape=(batch_size, 1)
	"""
	assert batch_size%(2*skip_window) == 0

	index_data = 0 #random.randint(0, len(data)-1)
	for step in range(num_steps):
		seq_index_batch = make_index_batch(batch_size, skip_window, index_data)		
		yield make_batch(data, seq_index_batch)
		
		# Update index
		index_data = (index_data + (batch_size//(2*skip_window)))%len(data)

def plot_with_labels(low_dim_embs, labels, fpath='tsne.png'):
	"""
	Function to visualise the result

	args:
		low_dim_embs:
		labes 		:
		fpath 		: str or pathlike object
	return:	
					: None
	"""
	assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
	
	plt.figure(figsize=(18, 18))  # in inches
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i, :]
		plt.scatter(x, y)
		plt.annotate(
					label,
					xy=(x, y),
					xytext=(5, 2),
					textcoords='offset points',
					ha='right',
					va='bottom')
	plt.savefig(fpath)

# Read data
vocab = read_data(FPATH_DATA)
vocab_size = 50000

print("Data size {}\n".format(len(vocab)))

data, dikt = build_dataset(vocab, vocab_size)
dikt_rvrs = reverse_dikt(dikt)
del vocab  # Hint to reduce memory.
write_metadata(dikt, "./logs/tsv.tsv")

# Hyperparameters
batch_size = 128
num_steps = 50000

dim_embed = 128 	# Dimension of the embedding vector.
num_sampled = 128 	# Number of negative examples to sample.
skip_window = 1 	# How many words to consider left and right.

test_size = 16 		# Size of random set of words to evaluate similarity on.
test_window = 100 	# Only pick samples in the head - more frequent - of the distribution.
test_examples = np.random.choice(test_window, test_size, replace=False)

##############################################################################
# Build Graph

graph = tf.Graph()
with graph.as_default():
	# Placeholders
	count_step = tf.Variable(0, name="count_step")
	
	inputs_train = tf.placeholder(tf.int32, shape=[batch_size])
	labels_train = tf.placeholder(tf.int32, shape=[batch_size, 1])
	
	test_dataset = tf.constant(test_examples, dtype=tf.int32)

	# Ops and variables pinned to the CPU because of missing GPU implementation
	with tf.device('/cpu:0'):
		# Look up embeddings for inputs.
		embeddings = tf.Variable(tf.random_uniform([vocab_size, dim_embed], -1.0, 1.0), name="embeddings")
		mat_embed = tf.nn.embedding_lookup(embeddings, inputs_train)

		# Construct the variables for the NCE loss
		W_nce = tf.Variable(tf.truncated_normal([vocab_size, dim_embed], stddev=1.0/math.sqrt(dim_embed)), name="W_nce")
		b_nce = tf.Variable(tf.zeros([vocab_size]), name="b_nce")

	# Compute the average NCE loss for the batch.
	# tf.nce_loss automatically draws a new sample of the negative labels each
	# time we evaluate the loss.
	loss = tf.reduce_mean(tf.nn.nce_loss(
										weights=W_nce,
										biases=b_nce,
										labels=labels_train,
										inputs=mat_embed,
										num_sampled=num_sampled,
										num_classes=vocab_size))

	# Construct the SGD optimizer using a learning rate of 1.0.
	optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

	# Compute the cosine similarity between minibatch examples and all embeddings.
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings/norm
	test_embeddings = tf.nn.embedding_lookup(normalized_embeddings, test_dataset)	
	similarity = tf.matmul(test_embeddings, normalized_embeddings, transpose_b=True)

	# Write summary for Tensorboard
	tf.summary.scalar('loss', loss)
	tf.summary.histogram('W_nce', b_nce)
	tf.summary.histogram('b_nce', b_nce)
	merged = tf.summary.merge_all()

	# Define saver to save/restore trained result
	saver = tf.train.Saver()

from tensorflow.contrib.tensorboard.plugins import projector                                                                                                                                                         

	


##############################################################################
# Run Session
with tf.Session(graph=graph) as sess:
	# Initialise the variables
	sess.run(tf.global_variables_initializer())
	print("Initialised")

	# Define summary writer
	summaries_dir = './logs'
	summary_writer = tf.summary.FileWriter(summaries_dir, sess.graph)

	

	# Load existing data (if any)		
	try:
		saver.restore(sess, FPATH_CKPT)
		print('Model restored')
	except tf.errors.NotFoundError: print('No saved model found')
	except tf.errors.InvalidArgumentError: print('Model structure has change. Rebuild model')

	# Training iterations
	average_loss = 0
	for step, (batch, labels) in enumerate(generate_batch(data, batch_size, skip_window, num_steps)):
		feed_dict = {inputs_train:batch, labels_train:labels}

		summ, _, loss_val = sess.run([merged, optimizer, loss], feed_dict=feed_dict)
		average_loss += loss_val

		summary_writer.add_summary(summ, count_step.eval()+step)

		if step%2000 == 0:
			if step > 0:
				average_loss /= 1000
			# The average loss is an estimate of the loss over the last 2000 batches.
			print("Average loss at step {0} : {1}".format(count_step.eval()+step, average_loss))
			average_loss = 0

		# Note that this is expensive (~20% slowdown if computed every 500 steps)
		if step%10000 == 0:
			sim = similarity.eval()
			for i in range(test_size):
				test_word = dikt_rvrs[test_examples[i]]
				top_k = 8  # number of nearest neighbors
				nearest = (-sim[i, :]).argsort()[1:top_k + 1]
				log_str = 'Nearest to %s:' % test_word
				for k in range(top_k): log_str = log_str + " " + dikt_rvrs[nearest[k]]
				print(log_str)
			
			# for i in range(batch_size): print('{0} {1} -> {2} {3}'.format(batch[i], dikt_rvrs[batch[i]], labels[i, 0], dikt_rvrs[labels[i, 0]]))

	config = projector.ProjectorConfig()

	embedding = config.embeddings.add()
	embedding.tensor_name = embeddings.name
	embedding.metadata_path = "./tsv.tsv"

	projector.visualize_embeddings(summary_writer, config)

	# Save the variables
	sess.run(count_step.assign(count_step.eval() + num_steps))
	saver.save(sess, FPATH_CKPT)
	print("Model saved in file: {0}".format(FPATH_CKPT))

	# Variable reaching out of the session for plot
	final_embeddings = normalized_embeddings.eval()

##############################################################################
# Visualize the embeddings 

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [dikt_rvrs[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)