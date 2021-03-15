# pyclglove

python + OpenCL module of the original standford **GloVe** word embeddings (https://github.com/stanfordnlp/GloVe)

### Motivation

I already wrote pyglove (https://github.com/otterrrr/pyglove) but this pure-python version is too slow to test even small-sized corpus. In this **pyclglove**, OpenCL version of pyglove, time to build Glove model is reduced much. In other words, you can take advantage of any devices supporting OpenCL for speed-up

### Installation

1. Download python package file: [pyclglove-0.1.0.tar.gz](https://github.com/otterrrr/pyclglove/blob/master/dist/pyclglove-0.1.0.tar.gz)
1. pip install pyclglove-0.1.0.tar.gz
   1. You might have to install pyopencl in advance when there's a dependency conflict

### Example

##### test_simple.py (enclosed)

```python
import pyclglove

sentences = [
    ['english', 'is', 'language'],
    ['korean', 'is', 'language'],
    ['apple', 'is', 'fruit'],
    ['orange', 'is', 'fruit']
]
glove = pyclglove.Glove(sentences, 5, verbose=True)

  [Initialization] parameters = {'num_component': 5, 'min_count': 1, 'max_vocab': 0, 'window_size': 15, 'distance_weighting': True, 'verbose': True}
  [Building Vocabulary] parameters = {'max_vocab': 0, 'verbose': True, 'min_count': 1}
  [Building Vocabulary] result = {'len(words)': 7, 'word[0]': ('is', 4), 'word[-1]': ('orange', 1)}
  [Counting Cooccurrence] parameters = {'window_size': 15, 'distance_weighting': True, 'verbose': True}
  [Counting Cooccurrence] result = {'len(cooccur_list)': 20, 'max(cooccur_list.count)': ((1, 0), 2.0), 'min(cooccur_list.count)': ((4, 2), 0.5)}

glove.fit(num_iteration=10, verbose=True)

  [Training Model] parameters = {'self': <pyclglove.Glove object at 0x0000022AC398D908>, 'force_initialize': False, 'num_iteration': 10, 'x_max': 100, 'alpha': 0.75, 'learning_rate': 0.05, 'verbose': True, 'num_procs': 8192}
  iteration # 0 ... loss = 0.000129
  iteration # 1 ... loss = 0.000115
  iteration # 2 ... loss = 0.000103
  iteration # 3 ... loss = 0.000094
  iteration # 4 ... loss = 0.000087
  iteration # 5 ... loss = 0.000080
  iteration # 6 ... loss = 0.000075
  iteration # 7 ... loss = 0.000070
  iteration # 8 ... loss = 0.000065
  iteration # 9 ... loss = 0.000061

print(glove.word_vector[glove.word_to_wid['language']])

  [[ 0.03173695 -0.13118016  0.05604964  0.04711236  0.08338707]
   [-0.03173398  0.07788379 -0.12542755  0.02645583  0.06843425]
   [ 0.12232077 -0.02306986  0.07213898 -0.14264075  0.09693842]
   [-0.01151574  0.05668031 -0.04010666 -0.04478449 -0.02876378]
   [ 0.0619041  -0.0040354  -0.02396944 -0.00720391 -0.06050783]
   [-0.11616069 -0.13800071 -0.02326163 -0.05045341  0.03023763]
   [ 0.04481338  0.00582148 -0.03377046 -0.12290663 -0.02710813]]

print(glove.word_to_wid)

  {'is': 0, 'fruit': 1, 'language': 2, 'apple': 3, 'english': 4, 'korean': 5, 'orange': 6}

print(glove.wid_to_word)

  {0: 'is', 1: 'fruit', 2: 'language', 3: 'apple', 4: 'english', 5: 'korean', 6: 'orange'}

print(glove.word_vector)

  [ 0.12232077 -0.02306986  0.07213898 -0.14264075  0.09693842]

```

### Note

OpenCL performance usually depends on devices and its DoP(Degree of Parallelism) setting suitable
* For popular GPU devices like NVIDIA Geforce and AMD Radeon series
  * More than thousands DoP works fine, e.g. 4096, 8192
* For CPU devices like Intel i-series or AMD A-series
  * Works better on small DoP close to the number of logical processors, e.g. 8, 12

One DoP parameter 'num_procs' can be provided into 'fit' function as follows
```
glove.fit(num_iteration=10, num_procs=4096) # default-value: 8192
```

### Limitation

Features the original stanford **Glove** supports but **pyglove** doesn't
* Memory-bound execution
  * The original **GloVe** implementation has memory-bound execution logic. In other words, it flushes out intermediate result over memory threshold
  * However, **pyglove** works assuming that system memory is sufficient to contain all the corpus and intermediate results
  * Hence, please make sure your system can provide enough memory
* Fixed parameters in function body
  * cooccurence_count.symmetric (fixed as True)
  * glove.word_vector.model (fixed as 3)
    1. result word_vectors consisting of target and context vectors including biases
    1. result word_vectors consisting of target vectors without biases
    1. result word_vectors consisting of target and context vectors without biases

Common limitation among all the **Glove** implementations including the original one
* Read-write or write-write conflict among several weight vector updates
  * Slightly different outcomes can be made even if you assigned a specific random seed but the difference isn't that significant

### Future items

* Performance improvement on counting cooccurrence, which performs only on pure-python now