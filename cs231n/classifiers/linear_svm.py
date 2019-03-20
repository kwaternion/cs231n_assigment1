import numpy as np
#from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      margin = scores[j] - correct_class_score + 1
      if j == y[i]:
        continue
      if margin > 0:
        loss += margin
        dW[:, y[i]] = dW[:, y[i]] - X[i]
        dW[:, j] = dW[:, j] + X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += 2 * reg * W

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_train = X.shape[0]
  scores = X.dot(W) #NxC
  corr_indices = (np.arange(scores.shape[0]), y)
  corr_score = scores[corr_indices] #N
  margins = scores - corr_score[:, np.newaxis] + 1 #NxC
  margins_maks = margins > 0
  margins_th = margins * margins_maks
  margins_th[corr_indices] = 0
  loss = margins_th.sum()/num_train
  loss += reg * np.sum(W * W)

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  num_margins = np.array(margins_maks, dtype="int")  # NxC
  num_margins[corr_indices] = 0
  num_incorrect = num_margins.sum(axis=1)  # N
  num_margins[corr_indices] = - num_incorrect
  dW = X.T @ num_margins   # W = DxC, X = N,D, num_margins = NxC
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################




  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
