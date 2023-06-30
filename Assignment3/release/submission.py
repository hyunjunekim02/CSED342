import shell
import util
import wordsegUtil



############################################################
# Problem 1: Word Segmentation

# Problem 1a: Solve the word segmentation problem under a unigram model

class WordSegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == len(self.query)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        result = []
        for i in  range(state + 1, len(self.query) + 1):
            word = self.query[state:i]
            result.append((word, i, self.unigramCost(word)))
        return result
        # END_YOUR_ANSWER

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch()
    ucs.solve(WordSegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

# Problem 1b: Solve the k-word segmentation problem under a unigram model

class KWordSegmentationProblem(util.SearchProblem):
    def __init__(self, k, query, unigramCost):
        self.k = k
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (0, self.k)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == (len(self.query), 0)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 7 lines of code, but don't worry if you deviate from this)
        result = []
        curr_i, words_left = state
        if words_left == 0: return result
        for i in range(curr_i + 1, len(self.query) + 1):
            word = self.query[curr_i:i]
            result.append((word, (i, words_left - 1), self.unigramCost(word)))
        return result
        # END_YOUR_ANSWER

def segmentKWords(k, query, unigramCost):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(KWordSegmentationProblem(k, query, unigramCost))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 2: Vowel Insertion

# Problem 2a: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.queryWords)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 9 lines of code, but don't worry if you deviate from this)
        result = []
        curr_i, prev_word = state
        if curr_i < len(self.queryWords):
            curr_word = self.queryWords[curr_i]
            for filled in self.possibleFills(curr_word) or [curr_word]:
                next_word = filled
                cost = self.bigramCost(prev_word, next_word)
                result.append((next_word, (curr_i + 1, next_word), cost))
        return result
        # END_YOUR_ANSWER

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

# Problem 2b: Solve the limited vowel insertion problem under a bigram cost

class LimitedVowelInsertionProblem(util.SearchProblem):
    def __init__(self, impossibleVowels, queryWords, bigramCost, possibleFills):
        self.impossibleVowels = impossibleVowels
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.queryWords)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 10 lines of code, but don't worry if you deviate from this)
        result = []
        curr_i, prev_word = state
        if curr_i < len(self.queryWords):
            curr_word = self.queryWords[curr_i]
            for filled in (self.possibleFills(curr_word) or [curr_word]):
                if not any(c in self.impossibleVowels for c in filled):
                    next_word = filled
                    cost = self.bigramCost(prev_word, next_word)
                    result.append((next_word, (curr_i + 1, next_word), cost))
        return result
        # END_YOUR_ANSWER

def insertLimitedVowels(impossibleVowels, queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(LimitedVowelInsertionProblem(impossibleVowels, queryWords, bigramCost, possibleFills))
    return ' '.join(w.replace(' ', '') for w in queryWords) if ucs.actions is None else ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 3: Putting It Together

# Problem 3a: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.query)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        result = []
        curr_i, prev_word = state
        for i in range(1, len(self.query) - curr_i + 1):
                curr_word = self.query[curr_i: curr_i + i]
                for filled in self.possibleFills(curr_word):
                    cost = self.bigramCost(prev_word, filled)
                    result.append((filled, (curr_i + i, filled), cost))
        return result
        # END_YOUR_ANSWER

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 4: A* search

# Problem 4a: Define an admissible but not consistent heuristic function

class SimpleProblem(util.SearchProblem):
    def __init__(self):
        # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
        self.start_state = "A"
        self.stateB = "B"
        self.stateC = "C"
        self.end_state = "D"
        # END_YOUR_ANSWER

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.start_state
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == self.end_state
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
        if state == self.start_state: return [("A->B", self.stateB, 3), ("A->C", self.stateC, 1)]
        elif state == self.stateB: return [("B->D", self.end_state, 1)]
        elif state == self.stateC: return [("C->B", self.stateB, 1)]
        # END_YOUR_ANSWER

def admissibleButInconsistentHeuristic(state):
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    heuristic = {"A": 3, "B": 0, "C": 2, "D": 0}
    return heuristic[state]
    # END_YOUR_ANSWER

# Problem 4b: Apply a heuristic function to the joint segmentation-and-insertion problem

def makeWordCost(bigramCost, wordPairs):
    """
    :param bigramCost: learned bigram cost from a training corpus
    :param wordPairs: all word pairs in the training corpus
    :returns: wordCost, which is a function from word to cost
    """
    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    Costsword = {}
    for w1, w2 in wordPairs:
        cost = bigramCost(w1, w2)
        if w2 not in Costsword or cost < Costsword[w2]:
            Costsword[w2] = cost
    return lambda w: Costsword.get(w, bigramCost(wordsegUtil.SENTENCE_UNK, w))
    # END_YOUR_ANSWER

class RelaxedProblem(util.SearchProblem):
    def __init__(self, query, wordCost, possibleFills):
        self.query = query
        self.wordCost = wordCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == len(self.query)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        result = []
        for step in range(1, len(self.query) - state + 1):
            next_state = state + step
            word = self.query[state:next_state]
            fills = self.possibleFills(word)
            for fill in fills:
                result.append((fill, next_state, self.wordCost(fill)))
        return result
        # END_YOUR_ANSWER

def makeHeuristic(query, wordCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    relaxed_problem = RelaxedProblem(query, wordCost, possibleFills)
    dp = util.DynamicProgramming(relaxed_problem)
    return lambda state: dp(state[0])
    # END_YOUR_ANSWER

def fastSegmentAndInsert(query, bigramCost, wordCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    heuristic = makeHeuristic(query, wordCost, possibleFills)
    ucs = util.UniformCostSearch()
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills), heuristic=heuristic)
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################

if __name__ == '__main__':
    shell.main()
