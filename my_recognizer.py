import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # Get the list of x, and lengths
    hwords = test_set.get_all_Xlengths()
    for word_id in range(0, len(test_set.get_all_sequences())):

        words_prob = {}
        best_score = float('-Inf')
        guess_word = None
        X, lengths = hwords[word_id]
        
        # for every word, we map the probability
        # and guess the best word
        for word, model in models.items():
            try:
                score = model.score(X, lengths)
            except:
                # set score to -inf if get score fails
                score = float('-Inf')
                
            words_prob[word] = score
            if score > best_score:
                best_score = score
                guess_word = word
                    
        probabilities.append(words_prob)
        guesses.append(guess_word)

    return probabilities, guesses
