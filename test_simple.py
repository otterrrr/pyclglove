import pyclglove

sentences = [
    ['english', 'is', 'language'],
    ['korean', 'is', 'language'],
    ['apple', 'is', 'fruit'],
    ['orange', 'is', 'fruit']
]
glove = pyclglove.Glove(sentences, 5, verbose=True)
glove.fit(num_iteration=10, verbose=True)
print(glove.word_vector)
print(glove.word_to_wid)
print(glove.wid_to_word)
print(glove.word_vector[glove.word_to_wid['language']])