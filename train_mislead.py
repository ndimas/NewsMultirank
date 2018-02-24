
# Import relevant packages and modules
from prediction.util import *
import random
import tensorflow as tf
import pickle 
from tqdm import tqdm as t

# Set file names
file_train_instances = "Training_Datasets/train_stances.csv"
file_train_bodies = "Training_Datasets/train_bodies.csv"
file_test_instances = "Training_Datasets/test_stances_unlabeled.csv"
file_test_bodies = "Training_Datasets/test_bodies.csv"

# Initialise hyperparameters
r = random.Random()
lim_unigram = 5000
target_size = 4
hidden_size = 100
train_keep_prob = 0.6
l2_alpha = 0.00001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 500
epochs = 50

# Load data sets
raw_train = FNCData(file_train_instances, file_train_bodies)
raw_test = FNCData(file_test_instances, file_test_bodies)
n_train = len(raw_train.instances)

# Process data sets
train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
    pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)


pickle.dump(bow_vectorizer, open("BoW.p", "wb"))
pickle.dump(tfreq_vectorizer, open("Tfreq.p", "wb"))
pickle.dump(tfidf_vectorizer, open("Tfidf.p", "wb"))

feature_size = len(train_set[0])
test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

# Define model

# Create placeholders
features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
stances_pl = tf.placeholder(tf.int64, [None], 'stances')
keep_prob_pl = tf.placeholder(tf.float32)

#Create constants
train_stances_tensor = tf.constant(train_stances)

# Infer batch size
batch_size = tf.shape(features_pl)[0]

# Define multi-layer perceptron
hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), keep_prob=keep_prob_pl)
logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size), keep_prob=keep_prob_pl)
logits = tf.reshape(logits_flat, [batch_size, target_size])

# Define L2 loss
tf_vars = tf.trainable_variables()
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

# Define overall loss
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits( labels = stances_pl , logits = logits) + l2_loss)

# Define prediction
softmaxed_logits = tf.nn.softmax(logits)
predict = tf.argmax(softmaxed_logits, 1)

# Train model


# Define optimiser
opt_func = tf.train.AdamOptimizer(learn_rate)
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
opt_op = opt_func.apply_gradients(zip(grads, tf_vars))



# Perform training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_pred = sess.run(predict, feed_dict = {features_pl: train_set, keep_prob_pl: 1.0})
    print(-1, " ", np.mean(np.equal(train_stances, train_pred)))

    for epoch in t(range(epochs)):
        total_loss = 0
        indices = list(range(n_train))
        r.shuffle(indices)

        for i in t(range(n_train // batch_size_train)):
            batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
            batch_features = [train_set[i] for i in batch_indices]
            batch_stances = [train_stances[i] for i in batch_indices]

            batch_feed_dict = {features_pl: batch_features, stances_pl: batch_stances, keep_prob_pl: train_keep_prob}
            _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
            total_loss += current_loss
            
     # Predict
        #train_pred = sess.run(predict, feed_dict = {features_pl: train_set, keep_prob_pl: 1.0})
        #print(epoch, " ", np.mean(np.equal(train_stances, train_pred)))

    saver = tf.train.Saver()
    saver.save(sess, './Pretrained_models/misleadmodel.ckpt')
