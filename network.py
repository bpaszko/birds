import tensorflow as tf
import numpy as np 
from PIL import Image

def table_to_image(tb):
    buffer = []
    for i in range(32):
        for j in range(32):
            buffer.append((tb[i][j][0], tb[i][j][1], tb[i][j][2]))
    out = Image.new('RGBA', (32, 32))
    out.putdata(buffer)
    out2 = out.resize((96,96))
    out2.show()
 
graph = tf.Graph()
image_size = 32
num_channels = 3
patch_size = 3
depth = 16
num_hidden = 512
num_labels = 2
batch_size = 64
f_size = (image_size-2)//2
s_size = (f_size-2)//2
t_size = s_size

def accuracy(pred, labels):
    return (100.0 * np.sum(np.argmax(pred, 1) == np.argmax(labels, 1))/pred.shape[0])

with graph.as_default():
    tf_train_data= tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels)) 
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_data= tf.constant(valid_data)
    tf_test_data= tf.constant(test_data)

    global_step = tf.Variable(0)
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, 2*depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.zeros([2*depth]))
    layer3_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, 2*depth, 4*depth], stddev=0.1))
    layer3_biases = tf.Variable(tf.zeros([4*depth]))
    layer4_weights=tf.Variable(tf.truncated_normal([(t_size)*(t_size)*(4*depth), num_hidden], stddev=0.1))
    layer4_biases = tf.Variable(tf.zeros([num_hidden]))
    layer5_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer5_biases = tf.Variable(tf.zeros([num_labels]))

    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.relu(conv + layer1_biases)
        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.relu(conv + layer2_biases)
        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        conv =  tf.nn.conv2d(pool, layer3_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer3_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]])
        print(tf.shape(reshape))
        hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights)+layer4_biases)
        hidden = tf.nn.dropout(hidden, 0.5)
        return tf.matmul(hidden, layer5_weights) + layer5_biases

    logits = model(tf_train_data)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    learning_rate = tf.train.exponential_decay(0.0001, global_step, 2000, 0.95, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_predictions = tf.nn.softmax(logits)
    valid_predictions = tf.nn.softmax(model(tf_valid_data))
    test_predictions = tf.nn.softmax(model(tf_test_data))

def learning_conv():
    num_steps = 112500
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized\n')
        for step in range(num_steps):
            offset = (step*batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_data[offset:(offset+batch_size), :, :, :]
            batch_labels= train_labels[offset:(offset+batch_size), :]
            feed_dict = {tf_train_data: batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_predictions], feed_dict=feed_dict)
            if step % 10000 == 0:
                print('Loss at step %d : %s' % (step, l))
                print('Accuracy on training data : %.2f' % accuracy(predictions, batch_labels))
                print('Accuracy on validation data : %.2f' % accuracy(valid_predictions.eval(), valid_labels))
                print('---------------------\n')
        print('Learning complete')
        print('Accuracy on test data : %.2f' % accuracy(test_predictions.eval(), test_labels))
                
