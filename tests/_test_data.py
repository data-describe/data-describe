import pandas as pd
import numpy as np

np.random.seed(22)
DATA = pd.DataFrame(
    {
        "a": np.random.normal(2, 1.2, size=250),
        "b": np.random.normal(3, 1.5, size=250),
        "c": np.random.normal(9, 0.2, size=250),
        "d": np.random.choice(["x", "y"], size=250),
        "e": np.random.choice(["v", "w"], p=[0.01, 0.99], size=250),
    }
)

TEXT_DATA = {
    'test_list_main':
    [
        "This is an article talking about Spider-man.",
        "Spiders are one of the weirdest things on earth.",
    ],

    'answer_key_tokenized':
    [
        ["This", "is", "an", "article", "talking", "about", "Spider-man", "."],
        ["Spiders", "are", "one", "of", "the", "weirdest", "things", "on", "earth", "."]
    ],

    'answer_key_lower':
    [
        ["this", "is", "an", "article", "talking", "about", "spider-man", "."],
        ["spiders", "are", "one", "of", "the", "weirdest", "things", "on", "earth", "."]
    ],

    'answer_key_remove_punct':
    [
        ['This', 'is', 'an', 'article', 'talking', 'about', 'Spider-man'],
        ['Spiders', 'are', 'one', 'of', 'the', 'weirdest', 'things', 'on', 'earth']
    ],

    'answer_key_remove_all_punct_no_space':
    [
        ['This', 'is', 'an', 'article', 'talking', 'about', 'Spiderman', ''],
        ['Spiders', 'are', 'one', 'of', 'the', 'weirdest', 'things', 'on', 'earth', '']
    ],

    'answer_key_remove_all_punct_with_space':
    [
        ['This', 'is', 'an', 'article', 'talking', 'about', 'Spider', 'man'],
        ['Spiders', 'are', 'one', 'of', 'the', 'weirdest', 'things', 'on', 'earth']
    ],

    'answer_key_remove_stop_words':
    [
        ['article', 'talking', 'spider-man', '.'],
        ['spiders', 'one', 'weirdest', 'things', 'earth', '.']
    ],

    'answer_key_remove_stop_words_more':
    [
        ['article', 'talking', '.'],
        ['one', 'weirdest', 'things', 'earth', '.']
    ],

    'test_list_digits':
    [["th3is", "is", "8", "a", "d1s4ster", "but", "no", "numb3rs"]],

    'answer_key_remove_digits':
    [["", "is", "", "a", "", "but", "no", ""]],

    'test_list_single_char_and_spaces':
    [["a", " ", "nice", "    ", "day", "to", "b", "outside", "thank you"]],

    'answer_key_single_char_and_spaces':
    [["nice", "day", "to", "outside", "thank", "you"]],

    'test_list_stopwords':
    [["please", "do", "not", "eat", "the", "rest", "of", "my", "pineapple", "pizza"]],

    'more_words':
    ['spiders', 'spider-man'],

    'answer_key_remove_stopwords':
    [["please", "eat", "rest", "pineapple", "pizza"]],

    'answer_key_remove_stopwords_more':
    [["please", "rest", "pineapple"]],

    'test_list_lem_and_stem':
    ["Mars is the greatest planet to start terraforming; it would be amazing to see geese flying!"],

    'answer_key_lem':
    [["mar", "greatest", "planet", "start", "terraforming", "would", "amazing", "see", "goose", "flying"]],

    'answer_key_stem':
    [["mar", "greatest", "planet", "start", "terraform", "would", "amaz", "see", "gees", "fly"]],

    'test_list_custom':
    ["this is an absolutely phenomenal day."],

    'answer_key_custom':
    [["THIS", "IS", "AN", "ABSOLUTELY", "PHENOMENAL", "DAY", "."]]
}
