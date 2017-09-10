from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from random import shuffle
import numpy as np
from reader import parseData
import time

seed = 128
rng = np.random.RandomState(seed)
model_path = "tempModelsCNNFolds/model"
best_classifier=None
mean_metrics = []

#X_, y_ = parseData()
#clf = joblib.load('model_classifier_positive_negative.pkl') 
X_, y_= parseData(isImage=True)

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot[np.arange(num_labels), labels_dense] = 1
    
    return labels_one_hot

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

def batch_creator(batch_size, dataset_length, y):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval('train_x')[[batch_mask]].reshape(-1, n_input)
    #batch_x = preproc(batch_x)
    

    batch_y = y[[batch_mask]]
    batch_y = dense_to_one_hot(batch_y)

    return batch_x, batch_y

### set all variables

# Parameters
learning_rate = 0.004
training_iters = 60
batch_size = 50
display_step = 10

# Network Parameters
n_input = 56*92 # 
n_classes = 10 # 
dropout = 0.75 # Dropout, probability to keep units
epochs=60

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 56, 92, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 14*23*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([14*23*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing the variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

skf = StratifiedKFold(n_splits=10, shuffle=True)
number = 0
precisions,accuracies,recalls,f1s = [],[],[],[]

start = time.time()
for train_index, test_index in skf.split(X_, y_):
    shuffle(test_index)
    shuffle(train_index)

    X_train, X_test = X_[train_index], X_[test_index]
    y_train, y_test = y_[train_index], y_[test_index]
    
    train_x = np.stack(X_train)
    print("Fold : ", number)

    with tf.Session() as sess:
        # create initialized variables
        sess.run(init)
        
        for epoch in range(epochs):
            avg_cost = 0
            avg_acc = 0
            total_batch = int(len(train_x)/batch_size)
            for i in range(total_batch):
                batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], y_train)
                acc, _, loss = sess.run([accuracy, optimizer, cost], feed_dict = {x: batch_x, y: batch_y, keep_prob: dropout})
                
                avg_cost += loss/total_batch
                avg_acc += acc/total_batch
            
                
            print("Iter " + str(epoch) + ", Minibatch Loss= " + \
                  "{:.6f}".format(avg_cost) + ", Training Accuracy= " + \
                  "{:.5f}".format(avg_acc))
            #print "Epoch:", (epoch+1), "cost =", "{:.10f}".format(avg_cost)
        
        print "\nTraining complete!"
        # Save model weights to disk
        save_path = saver.save(sess, model_path +str(number) + ".ckpt")
        print("Model saved in file: %s" % save_path)

        # find predictions on val set
        predict = tf.argmax(pred, 1)
        predictions = predict.eval({x: X_test.reshape(-1, n_input), keep_prob: 1.})
        
        accuracies.append(accuracy_score(y_test, predictions))
        precisions.append(precision_score(y_test, predictions, average='weighted'))
        recalls.append(recall_score(y_test, predictions, average='weighted'))
        f1s.append(f1_score(y_test, predictions, average='weighted'))
        print("accuracy : ", accuracy_score(y_test, predictions ) )
        print("precision : ", precision_score(y_test, predictions, average='weighted' ) )
        print("recall : ", recall_score(y_test, predictions,average='weighted' ) )
        print("f1 : ", f1_score(y_test, predictions,average='weighted' ) )
        print("\n")
    
    number+=1

print("accuracy avg : ", np.mean(accuracies) )
print("precision avg : ", np.mean(precisions) )
print("recall avg : ", np.mean(recalls) )
print("f1 avg : ", np.mean(f1s) )

print("it took", time.time() - start, "seconds.")