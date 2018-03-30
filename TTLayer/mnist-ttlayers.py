import t3f

from utils import TTDense
  

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# prepare data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

xs = tf.placeholder(tf.float32, [None, 784]) # 28*28*1
ys = tf.placeholder(tf.float32, [None, 10])  # 0~9

 
image = tf.reshape(xs, [-1, 28, 28, 1])


'''
# the model of the fully-connected network
weights = tf.Variable(tf.random_normal([784, 10]))

biases = tf.Variable(tf.zeros([1, 10]))

outputs = tf.matmul(xs, weights) + biases

'''



l = tf.layers.conv2d(image, 32, 3, padding='same', activation=tf.nn.relu, name='conv0')

l = tf.layers.max_pooling2d(l, 2, 2, padding='valid')

l = tf.layers.conv2d(l, 32, 3, padding='same', activation=tf.nn.relu, name='conv1')

l = tf.layers.conv2d(l, 32, 3, padding='same', activation=tf.nn.relu, name='conv2')

l = tf.layers.max_pooling2d(l, 2, 2, padding='valid')

l = tf.layers.conv2d(l, 32, 3, padding='same', activation=tf.nn.relu, name='conv3')

print("----------------------")
print(l)
l = tf.layers.flatten(l)

#l = tf.layers.dense(l, 256, activation=tf.nn.relu, name='fc0')

l = TTDense(row_dims=[7, 7, 32, 1], column_dims=[1, 256, 1, 1], tt_rank=16)(l)
#l = tf.layers.dropout(l, rate=0.5,training=True)

#logits = tf.layers.dense(l, 10, activation=tf.identity, name='fc1')

        

logits = TTDense(row_dims=[256, 1, 1, 1], column_dims=[1, 2, 5, 1], tt_rank=16)(l)


# prediction
predictions = tf.nn.softmax(logits)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(predictions),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# compute the accuracy
correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(ys, 1))

print (correct_predictions)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)

        sess.run(train_step, feed_dict={xs: batch_xs,ys: batch_ys})

        if i % 50 == 0:
            
            print("steps : %d " %i,"accuracy: ",sess.run(accuracy, feed_dict={
                xs: mnist.test.images,
                ys: mnist.test.labels
            }))
