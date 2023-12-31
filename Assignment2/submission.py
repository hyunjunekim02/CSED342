#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *


############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, touching, quite, impressive, not, boring
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'so': 1, 'touching':1, 'quite': 0, 'impressive': 0, 'not': -1, 'boring': -1}
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    words = x.split()
    featurevec = {}
    for word in words:
        if word not in featurevec: featurevec[word] = 0
        featurevec[word]+=1
    return featurevec
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    for t in range(numIters):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            score = 0
            for f in phi:
                if f in weights:
                    score += weights[f]*phi[f]
            prob = sigmoid(score)
            for f in phi:
                if y==1: gradient = phi[f]*(1-prob)
                else: gradient = -phi[f]*prob
                weights[f] = weights.get(f,0) + eta*gradient
    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: ngram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER (our solution is 12 lines of code, but don't worry if you deviate from this)
    words = x.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ""
        for j in range(n):
            ngram += words[i+j]
            if j < n-1:
                ngram += " "
        ngrams.append(ngram)
    phi = {}
    for ngram in ngrams:
        phi[ngram] = phi.get(ngram, 0) + 1
    # END_YOUR_ANSWER
    return phi

############################################################
# Problem 3a: k-means exercise
############################################################

def problem_3a_1():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -2, 'mu_y': 0}, {'mu_x': 3, 'mu_y': 0})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'mu_x': -0.5, 'mu_y': 1.5}, {'mu_x': 3, 'mu_y': 1.5}
    # END_YOUR_ANSWER

def problem_3a_2():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -1, 'mu_y': -1}, {'mu_x': 2, 'mu_y': 3})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'mu_x': -1, 'mu_y': 0}, {'mu_x': 2, 'mu_y': 2}
    # END_YOUR_ANSWER

############################################################
# Problem 3: k-means implementation
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_ANSWER (our solution is 40 lines of code, but don't worry if you deviate from this)
    centroids = random.sample(examples, K)
    example_feat = [set(example.keys()) for example in examples]

    is_converged = False
    for iteration in range(maxIters):

        assignments = []
        for example, features in zip(examples, example_feat):
            min_dist = math.inf
            closest_centroid_index = None
            for j, centroid in enumerate(centroids):
                distance = sum((example.get(f, 0) - centroid.get(f, 0)) ** 2 for f in features)
                if distance < min_dist:
                    min_dist = distance
                    closest_centroid_index = j
            assignments.append(closest_centroid_index)

        new_centroids = [{} for _ in range(K)]
        counts = [0] * K
        for example, assignment in zip(examples, assignments):
            for feature, value in example.items():
                new_centroids[assignment][feature] = new_centroids[assignment].get(feature, 0) + value
            counts[assignment] += 1
        for j in range(K):
            if counts[j] > 0:
                for feature in new_centroids[j]:
                    new_centroids[j][feature] /= counts[j]
            else:
                new_centroids[j] = centroids[j]

        if all(new_centroids[j] == centroids[j] for j in range(K)):
            is_converged = True
        else:
            centroids = new_centroids
        if is_converged:
            break

    loss = sum(sum((examples[i].get(f, 0) - centroids[assignments[i]].get(f, 0)) ** 2 for f in examples[i]) for i in range(len(examples)))
    return centroids, assignments, loss
    # END_YOUR_ANSWER
