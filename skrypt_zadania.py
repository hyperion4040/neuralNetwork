# Klasyfikacja punktow(obrazow) nalezacych do prostokata
#
#       (1,3) o----------------o (4,3)
#             |                |
#             |                |
#             |                |
#       (1,1) o----------------o (4,1)
#
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


# tf.disable_v2_behavior()
print(tf.__version__)
mnist = tf.keras.datasets.mnist


num_of_features = 2
num_of_epochs = 1500
num_to_show = 50
batch_size = 100

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])  # place for input vectors
y = tf.placeholder(tf.float32, shape=[None, 10])               # place for desired output of ANN

W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# IW = tf.Variable(tf.truncated_normal([num_of_features, nneu[0]],stddev=0.1))  # 1-st level weights initialized with normal distribution
# b1 = tf.Variable(tf.constant(0.1, shape=[nneu[0]])) # 1-st level biases -||-

# my_tensor = tf.constant([.........],dtype=float)
# IW_a = tf.Variable(my_tensor)                                # 1-st level weights - analytical version
# b1_a = tf.Variable(tf.constant([........],dtype=float))      # 1-st level biases -||-
#
# h1_a = .................................                     # analytical version
# h1 = ..................................                      # output values from 1-st level
#
#
# LW21 = ..................           # 2-nd level weights values
# b2 = ....................                                          # 2-nd level bias values
#
# LW21_a = ..................           # 2-nd level weights values - analytical version
# b2_a = ....................

# LW32 = .................           # maybe 3-d layer
# b3 = ...................

hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

# y_a = 1                # output flom ANN - analytical version
# y = hidden_out                   # output from ANN (single value using sigmoidal act.funct in range (0,1))



  # training method, step value, loss function
                                                                              # You can choose loss function

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

init_op = tf.global_variables_initializer()

y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))
# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)
# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




# the training process:
# ........................................................
# ........................................................

# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   total_batch = int(len(mnist.train.labels) / batch_size)
   for epoch in range(num_of_epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy],
                         feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))



# drawing decision boundary:
X1, X2 = np.meshgrid(np.linspace(0, 8, 120), np.linspace(0,8, 120))  # grid of points in 2D plane
P = np.stack((X1.flatten(),X2.flatten()), axis=1)                    # points formated for ANN input
Y = sess.run(y_a, feed_dict={x: P})                                    # ANN outputs for flatten grid  points
Z = np.reshape(Y,X1.shape)                                           # reshaping to shape of grid
plt.contourf(X1, X2, Z, levels=[0.5, 1.0])                           # curve for level=0.5 - a decision boundary, shaded class 1 area
plt.title('analytical method')
plt.show()


# drawing 3D mesh:
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap=cm.viridis)
plt.title('analytical method')
plt.show()


# drawing 3D mesh:
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
Y = sess.run(y, feed_dict={x: P})                                    # ANN outputs for flatten grid  points
Z = np.reshape(Y,X1.shape)
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap=cm.viridis)
plt.title('learning method')
plt.show()