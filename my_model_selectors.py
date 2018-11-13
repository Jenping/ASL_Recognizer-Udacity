import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # initialize best score and model
        best_score = float("inf")
        best_model = None
        
        # loop through all the components 
        for num_state in range(self.min_n_components, self.max_n_components):
            try:
                model = GaussianHMM(n_components=num_state, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                
                # Using the BIC equations
                parameters = num_state * num_state + 2 * num_state * len(self.X[0]) - 1
                bic = (-2) * logL + math.log(len(self.X)) * parameters
                
                if bic < best_score:
                    best_score = bic
                    best_model = model
            except:
                pass
        return best_model if best_model is not None else self.base_model(self.n_constant)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

         # TODO implement model selection based on DIC scores
        # initialize best score and model
        best_score = float("-inf")
        best_model = None
        
        # loop through all the components 
        for num_state in range(self.min_n_components, self.max_n_components):
            try:
                model = GaussianHMM(n_components=num_state, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                
                total_logL = 0.0
                M = 0
                for word in self.hwords:
                    if word != self.this_word:
                        X, lengths = self.hwords[word]
                        total_logL += model.score(X, lengths)
                        M += 1
                
                dic = logL - total_logL / M
                
                if dic > best_score:
                    best_score = dic
                    best_model = model
            except:
                pass
        return best_model if best_model is not None else self.base_model(self.n_constant)



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # initialize best score and model
        best_score = float("-inf")
        best_model = None
        logL_list = []
        
        # if there is not enough samples to do kfold split
        if (len(self.sequences) < 2):
            return self.base_model(self.n_constant)
        
        # loop through all the components 
        split_method = KFold(n_splits=min(3,len(self.sequences)))
        for num_state in range(self.min_n_components, self.max_n_components):
            # split the word sequences in to training and testing set
            # using KFold split
            try:
                for train_idx, test_idx in split_method.split(self.sequences):
                    X_train, lengths_train = combine_sequences(train_idx, self.sequences)
                    model = GaussianHMM(n_components=num_state, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                    X_test, lengths_test = combine_sequences(test_idx, self.sequences)
                    logL = model.score(X_test, lengths_test)
                    logL_list.append(logL)
                    
                mean_score = np.mean(logL_list)
    
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
            except:
                pass
            
        return best_model if best_model is not None else self.base_model(self.n_constant)
