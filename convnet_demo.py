

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

slim = tf.contrib.slim

mnist = input_data.read_data_sets('./tmp/data/', one_hot=True)

learning_rate = 0.001
num_steps = 500
batch_size = 128

num_input = 784
num_classes = 10

sess = tf.Session()

dataset = tf.contrib.data.Dataset.from_tensor_slices(
    (mnist.train.images, mnist.train.labels))
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={})


X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
# keep_prob = tf.placeholder(tf.float32)

# def conv2d(x, W, b, strides=1):
#   x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
#   x = tf.nn.bias_add(x, b)
#   return tf.nn.relu(x)
#
# def maxpool2d(x, k=2):
#   return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
#                         padding='SAME')
#
# def conv_net(x, weights, biases, dropout):
#   x = tf.reshape(x, shape=[-1, 28, 28, 1])
#   conv1 = conv2d(x, weights['wc1'], biases['bc1'])
#   conv1 = maxpool2d(conv1, k=2)
#
#   conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
#   conv2 = maxpool2d(conv2, k=2)
#
#   fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
#   fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
#   fc1 = tf.nn.relu(fc1)
#
#   fc1 = tf.nn.dropout(fc1, dropout)
#   out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
#
#   return out

def slim_conv_net(x):
  x = tf.reshape(x, shape=[-1, 28, 28, 1])
  conv1 = slim.conv2d(x, 32, [5, 5], padding='SAME',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      activation_fn=tf.nn.relu, scope='conv1')
  conv1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
  conv2 = slim.conv2d(conv1, 64, [5, 5], padding='SAME',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                      activation_fn=tf.nn.relu, scope='conv2')
  conv2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
  conv2 = slim.flatten(conv2)
  fc1 = slim.fully_connected(conv2, 1024, activation_fn=None, scope='fc1')
  fc1 = slim.dropout(fc1, 0.5, scope='dropout')
  out = slim.fully_connected(fc1, num_classes, activation_fn=None, scope='out')
  return out

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# logits = conv_net(X, weights, biases, keep_prob)
logits = slim_conv_net(X)
prediction  = tf.nn.softmax(logits)

# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))

slim.losses.softmax_cross_entropy(logits, Y)

total_loss = slim.losses.get_total_loss()

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = slim.learning.create_train_op(total_loss, optimizer)

logdir = './slim_log'

slim.learning.train(
    train_op,
    logdir,
    number_of_steps=500,
    save_summaries_secs=10,
    save_interval_secs=10)
# train_op = optimizer.minimize(loss_op)

# correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init = tf.global_variables_initializer()

# with tf.Session() as sess:
#
#   # sess.run(init)
#
#   for step in range(1, num_steps+1):
#     batch_x, batch_y = mnist.train.next_batch(batch_size)
#     sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
#     if step % display_step == 0 or step == 1:
#       loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
#                                                            Y: batch_y})
#       print('Step ' + str(step) + ', Minibatch Loss= ' + \
#             '{:.4f}'.format(loss) + ', Training Accuracy= ' + \
#             '{:.3f}'.format(acc))
#   print('Optimization Finished!')
#   print('Test Accuracy:', \
#         sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
#                                       Y: mnist.test.labels[:256]}))
