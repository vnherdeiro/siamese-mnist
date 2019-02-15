'''Train a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import random
import click


from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Reshape
from keras.optimizers import RMSprop
from keras import backend as K


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

class Siamese4MNIST:

    def __init__(self, debug, export, epochs):
        self.debug = debug
        self.export = export
        self.epochs = epochs
        self.model = None
        self.base_network = None

    @staticmethod
    def create_base_network(input_dim,output_dim):
        '''Base network to be shared (eq. to feature extraction).
        '''
        model = Sequential()
        model.add( Reshape((28,28,1), input_shape=(784,)))
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(28,28,1)))
        model.add(Conv2D(64, (3, 3), activation='tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim, activation='tanh'))
        return model

    def build_model(self):
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        if self.debug:
            x_train, y_train = x_train[:1000], y_train[:1000]
        x_test = x_test.reshape(10000, 784)
        if self.debug:
            x_test, y_test = x_test[:1000], y_test[:1000]
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        self.x_train = x_train
        self.x_test = x_test
        input_dim = 784
        output_dim = 2

        # create training+test positive and negative pairs
        train_digit_indices = [np.where(y_train == i)[0] for i in range(10)]
        self.train_digit_indices = train_digit_indices
        tr_pairs, tr_y = create_pairs(x_train, train_digit_indices)
        self.tr_pairs, self.tr_y = tr_pairs, tr_y
        test_digit_indices = [np.where(y_test == i)[0] for i in range(10)]
        self.test_digit_indices = test_digit_indices
        te_pairs, te_y = create_pairs(x_test, test_digit_indices)
        self.te_pairs, self.te_y = te_pairs, te_y

        # network definition
        base_network = self.create_base_network(input_dim, output_dim)
        self.base_network = base_network

        input_a = Input(shape=(input_dim,))
        input_b = Input(shape=(input_dim,))

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])

        model = Model([input_a, input_b], distance)

        model.compile(loss=contrastive_loss, optimizer='adam')
        self.model = model

    def train_model(self):
        epochs = self.epochs if not self.debug else 1
        self.model.fit([self.tr_pairs[:, 0], self.tr_pairs[:, 1]], self.tr_y,
                  batch_size=128,
                  epochs=epochs,
                  validation_data=([self.te_pairs[:, 0], self.te_pairs[:, 1]], self.te_y))

    def export_model(self):
        if self.export and not self.debug:
            print( 'Exporting to disk')
            self.model.save('mnist_siamese.h5')

    def benchmark_model(self):
        # compute final accuracy on training and test sets
        pred = self.model.predict([self.tr_pairs[:, 0], self.tr_pairs[:, 1]])
        tr_acc = compute_accuracy(pred, self.tr_y)
        pred = self.model.predict([self.te_pairs[:, 0], self.te_pairs[:, 1]])
        te_acc = compute_accuracy(pred, self.te_y)

        print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
        print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    def visualize_model(self):
        '''
        VISUALIZATION
        TO DO: qualitative analysis of a confusion matrix from the 2d display
        '''
        plt.figure( figsize=(7,7))
        cm = plt.get_cmap('viridis')
        colors = iter( cm( np.linspace(0,1,10)))
        for digit in range(10):
            # train_images = x_train[ train_digit_indices[digit]]
            # train_vector = base_network.predict( train_images)
            # train_vector = np.mean( train_vector, axis=0)
            # test_images = x_test[ test_digit_indices[digit]]
            # test_vector = base_network.predict( test_images)
            # test_vector = np.mean( test_vector, axis=0)
            # print( train_vector, test_vector)
            color = next(colors)
            # plt.quiver( 0,0 , *train_vector, label='%d train'%digit, color=color)
            # plt.quiver( 0,0, *test_vector, label='%d test'%digit, color=color)
            # plt.plot( *train_vector.reshape(2,1), label='%d train'%digit, color=color, marker=r'$%d$'%digit, markersize=22, ls='none')
            # plt.plot( *test_vector.reshape(2,1), label='%d train'%digit, color=color, marker=r'$%d$'%digit, markersize=22, ls='none')

            # running for train, test concatenation (because we checked similarity)
            images = np.concatenate( (self.x_train[ self.train_digit_indices[digit]], self.x_test[ self.test_digit_indices[digit]]), axis=0)
            vector = self.base_network.predict( images, batch_size=128)
            # vector = np.mean( vector, axis=0)
            # plt.plot( *vector.reshape(2,1), label='%d'%digit, color=color, marker=r'$%d$'%digit, markersize=22, ls='none')
            # PLOTTING DISTRIBUTION INSTEAD
            x, y = vector.T
            sns.kdeplot(x, y, shade=True, shade_lowest=False, color=color, label=digit)
        plt.legend(ncol=2)
        plt.xticks( [])
        plt.yticks( [])
        plt.tight_layout()
        plt.show()


    def __call__(self):
        self.build_model()
        self.train_model()
        self.export_model()
        self.benchmark_model()
        self.visualize_model()


@click.command()
@click.option('--epochs', default=10, help='Number of epochs.')
@click.option('--save', is_flag=True)
@click.option('--debug', is_flag=True)
def main(epochs, save, debug):
    model = Siamese4MNIST(debug, save, epochs)
    model()

if __name__ == '__main__':
    main()
