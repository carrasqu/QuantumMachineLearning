import tensorflow as tf
import input_data
import sys

L=200
lx=4 #=int(raw_input('lx'))
V4d=lx*lx*lx*L # 4d volume

training=10000  #=int(raw_input('training'))
bsize=400 #=int(raw_input('bsize'))

# how does the data look like
Ntemp=104 #int(raw_input('Ntemp'))   #20 # number of different temperatures used in the simulation
samples_per_T=80  #int(raw_input('samples_per_T'))  #250 # number of samples per temperature value
samples_per_T_test=20 # int(raw_input('samples_per_T'))  #250 # number of samples per temperature value


numberlabels=2
mnist = input_data.read_data_sets(numberlabels,lx,'txt', one_hot=True)

print "reading sets ok"

#sys.exit("pare aqui")

# defining weighs and initlizatinon
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# defining the convolutional and max pool layers
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# defining the model

x = tf.placeholder("float", shape=[None, V4d])
y_ = tf.placeholder("float", shape=[None, numberlabels])

#first fully connected layer 
nlayer1=400
W_1 = weight_variable([V4d, nlayer1])
b_1 = bias_variable([nlayer1])


h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

# Dropout: To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.

keep_prob = tf.placeholder("float")
h_1_drop = tf.nn.dropout(h_1, keep_prob)

nlayer2=400
W_2 = weight_variable([nlayer1,nlayer2])
b_2 = bias_variable([nlayer2])


h_2 = tf.nn.relu(tf.matmul(h_1_drop, W_2) + b_2)

# Dropout: To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.

#keep_prob = tf.placeholder("float")
h_2_drop = tf.nn.dropout(h_2, keep_prob)


nlayer3=100
W_3 = weight_variable([nlayer2,nlayer3])
b_3 = bias_variable([nlayer3])

h_3 = tf.nn.relu(tf.matmul(h_2_drop, W_3) + b_3)

# Dropout: To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.

#keep_prob = tf.placeholder("float")
h_3_drop = tf.nn.dropout(h_3, keep_prob)


# readout layer. Finally, we add a softmax layer, just like for the one layer softmax regression above.

# weights and bias
W_fc4 = weight_variable([nlayer3, numberlabels])
b_fc4 = bias_variable([numberlabels])

# apply a softmax layer
y_conv=tf.nn.softmax(tf.matmul(h_3_drop, W_fc4) + b_fc4)


#Train and Evaluate the Model
# cost function to minimize
lamb=0.00001
#lamb=0.001
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0))) +lamb*(tf.nn.l2_loss(W_1)+tf.nn.l2_loss(W_2) )+lamb*(tf.nn.l2_loss(W_fc4)+tf.nn.l2_loss(W_3) ) 

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(training):
  batch = mnist.train.next_batch(bsize)
  if i%100 == 0:
    train_accuracy = sess.run(accuracy,feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
    print "test accuracy %g"%sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
#  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print "test accuracy %g"%sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


#producing data to get the plots we like

f = open('nnout.dat', 'w')

#output of neural net
ii=0
for i in range(Ntemp):
  av=0.0
  for j in range(samples_per_T_test):
        batch=(mnist.test.images[ii,:].reshape((1,V4d)),mnist.test.labels[ii,:].reshape((1,numberlabels)))
        res=sess.run(y_conv,feed_dict={x: batch[0], y_: batch[1],keep_prob: 1.0})
        av=av+res
        #print ii, res
        ii=ii+1
  av=av/samples_per_T_test
  f.write(str(i)+' '+str(av[0,0])+' '+str(av[0,1])+"\n")
f.close()


f = open('acc.dat', 'w')

# accuracy vs temperature
for ii in range(Ntemp):
  batch=(mnist.test.images[ii*samples_per_T_test:ii*samples_per_T_test+samples_per_T_test,:].reshape(samples_per_T_test,V4d), mnist.test.labels[ii*samples_per_T_test:ii*samples_per_T_test+samples_per_T_test,:].reshape((samples_per_T_test,numberlabels)) )
  train_accuracy = sess.run(accuracy,feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
  f.write(str(ii)+' '+str(train_accuracy)+"\n")
f.close()
 
  
