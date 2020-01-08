import tensorflow as tf
import numpy as np


def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images, numpy array of shape (N, H, W, 3)
    - y: Labels for X, numpy of shape (N,)
    - model: A SqueezeNet model that will be used to compute the saliency map.

    Returns:
    - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
    input images.
    """
    saliency = None
    # Compute the score of the correct class for each example.
    # This gives a Tensor with shape [N], the number of examples.
    #
    # Note: this is equivalent to scores[np.arange(N), y] we used in NumPy
    # for computing vectorized losses.

    ###############################################################################
    # TODO: Produce the saliency maps over a batch of images.                     #
    #                                                                             #
    # 1) Define a gradient tape object and watch input Image variable             #
    # 2) Compute the “loss” for the batch of given input images.                  #
    #    - get scores output by the model for the given batch of input images     #
    #    - use tf.gather_nd or tf.gather to get correct scores                    #
    # 3) Use the gradient() method of the gradient tape object to compute the     #
    #    gradient of the loss with respect to the image                           #
    # 4) Finally, process the returned gradient to compute the saliency map.      #
    ###############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = y.shape[0]
    tf_x = tf.Variable(X)
    with tf.GradientTape() as t:
        t.watch(tf_x)
        s = model(tf_x)
        scores = tf.gather_nd(s, tf.stack((tf.range(N), y), axis=1))
        dx = t.gradient(scores,tf_x)
        saliency = tf.reduce_max(dx, axis=3)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency


# def make_fooling_image(X, target_y, model):
#     """
#     Generate a fooling image that is close to X, but that the model classifies
#     as target_y.
#
#     Inputs:
#     - X: Input image, a numpy array of shape (1, 224, 224, 3)
#     - target_y: An integer in the range [0, 1000)
#     - model: Pretrained SqueezeNet model
#
#     Returns:
#     - X_fooling: An image that is close to X, but that is classifed as target_y
#     by the model.
#     """
#
#     # Make a copy of the input that we will modify
#     X_fooling = X.copy()
#
#     # Step size for the update
#     learning_rate = 1
#
#     ##############################################################################
#     # TODO: Generate a fooling image X_fooling that the model will classify as   #
#     # the class target_y. Use gradient *ascent* on the target class score, using #
#     # the model.scores Tensor to get the class scores for the model.image.   #
#     # When computing an update step, first normalize the gradient:               #
#     #   dX = learning_rate * g / ||g||_2                                         #
#     #                                                                            #
#     # You should write a training loop, where in each iteration, you make an     #
#     # update to the input image X_fooling (don't modify X). The loop should      #
#     # stop when the predicted class for the input is the same as target_y.       #
#     #                                                                            #
#     # HINT: Use tf.GradientTape() to keep track of your gradients and            #
#     # use tape.gradient to get the actual gradient with respect to X_fooling.    #
#     #                                                                            #
#     # HINT 2: For most examples, you should be able to generate a fooling image  #
#     # in fewer than 100 iterations of gradient ascent. You can print your        #
#     # progress over iterations to check your algorithm.                          #
#     ##############################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     X_fooling = tf.convert_to_tensor(X_fooling)
#     best_score = -1
#     i=0
#     target_score = model.scores[0][target_y]
#     g = tf.gradient(target_score, model.image)
#     updated = model.image + learning_rate * g/tf.norm(g,ord=2)
#
#     with get_session() as sess:
#
#         while i<100:
#             i=i+1
#             print(i)
#             with tf.GradientTape() as t:
#                 t.watch(X_fooling)
#                 scores = model(X_fooling)
#                 best_score = tf.argmax(scores[0])
#                 print(tf.reduce_all(tf.equal(best_score, target_y)))
#                 target_score = scores[0, target_y]
#                 g =
#                 dx = learning_rate * g/tf.norm(g, ord=2)
#                 X_fooling -= dx
#
#
#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     ##############################################################################
#     #                             END OF YOUR CODE                               #
#     ##############################################################################
#     return X_fooling


def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image, a numpy array of shape (1, 224, 224, 3)
    - target_y: An integer in the range [0, 1000)
    - model: Pretrained SqueezeNet model

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """

    # Make a copy of the input that we will modify
    X_fooling = X.copy()

    # Step size for the update
    learning_rate = 1

    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. Use gradient *ascent* on the target class score, using #
    # the model.scores Tensor to get the class scores for the model.image.   #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop, where in each iteration, you make an     #
    # update to the input image X_fooling (don't modify X). The loop should      #
    # stop when the predicted class for the input is the same as target_y.       #
    #                                                                            #
    # HINT: Use tf.GradientTape() to keep track of your gradients and            #
    # use tape.gradient to get the actual gradient with respect to X_fooling.    #
    #                                                                            #
    # HINT 2: For most examples, you should be able to generate a fooling image  #
    # in fewer than 100 iterations of gradient ascent. You can print your        #
    # progress over iterations to check your algorithm.                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X_fooling = tf.convert_to_tensor(X_fooling)

    best_score = -1
    while best_score!=target_y:
        with tf.GradientTape() as t:
            t.watch(X_fooling)
            scores = model(X_fooling)
            best_score = np.argmax(scores[0])
            target_score = scores[0, target_y]
            g = t.gradient(target_score, X_fooling)
            dx = learning_rate * g / tf.norm(g, ord=2)
            X_fooling += dx

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling

import sys
sys.path.append('../')
import time, os, json

from cs231n.classifiers.squeezenet import SqueezeNet
from cs231n.data_utils import load_tiny_imagenet
from cs231n.image_utils import preprocess_image, deprocess_image
from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD


SAVE_PATH = 'datasets/squeezenet.ckpt'

if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")

model = SqueezeNet()
status = model.load_weights(SAVE_PATH)

model.trainable = False


from cs231n.data_utils import load_imagenet_val
X_raw, y, class_names = load_imagenet_val(num=5, imagenet_fn='datasets/imagenet_val_25.npz')



X = np.array([preprocess_image(img) for img in X_raw])


idx = 0
Xi = X[idx][None]
target_y = 6
X_fooling = make_fooling_image(Xi, target_y, model)