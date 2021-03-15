cl_source = \
"""

__kernel void update_gradient_immediate(
    const unsigned int             L,     // number of words
    const unsigned int             Dim,   // dimension of a word
    const double                    Xmax,  // reduce effect of value over Xmax
    const double                    Alpha, // Y of pow function if value > Xmax
    const double                    LR,    // initial learning rate
    __global const unsigned int* restrict coo_group, // coo_group[group-id] = coo-list-range(begin,end)
    __global const unsigned int* restrict wids_target,
    __global const unsigned int* restrict wids_context,
    __global const double* restrict values,
    __global double* restrict W,
    __global double* restrict B,
    __global double* restrict Gw,
    __global double* restrict Gb,
    __global double* restrict coo_cost
    )
{
    const unsigned int gid = get_global_id(0);
    const unsigned int coo_range[2] = { coo_group[gid], coo_group[gid+1] };
    double cost = 0.0;
    unsigned int count = 0;
    unsigned int coo;
    for (coo = coo_range[0] ; coo < coo_range[1] ; ++coo)
    {
        const unsigned int wid_target = wids_target[coo];
        const unsigned int wid_context = wids_context[coo];
        double value = values[coo];
        double diff = 0.0;
        double fdiff = 0.0;
        double grad_w[2] = { 0.0, 0.0 };
        double upd_w[2] = { 0.0, 0.0 };
        double upd_b[2] = { 0.0, 0.0 };
        __global double* W0 = W;
        __global double* W1 = W + L*Dim;
        __global double* B0 = B;
        __global double* B1 = B + L;
        __global double* Gw0 = Gw;
        __global double* Gw1 = Gw + L*Dim;
        __global double* Gb0 = Gb;
        __global double* Gb1 = Gb + L;
        __global double* W0t = W0 + wid_target*Dim;
        __global double* W1t = W1 + wid_target*Dim;
        __global double* W0c = W0 + wid_context*Dim;
        __global double* W1c = W1 + wid_context*Dim;
        __global double* B0t = B0 + wid_target;
        __global double* B1c = B1 + wid_context;
        __global double* Gw0t = Gw0 + wid_target*Dim;
        __global double* Gw1c = Gw1 + wid_context*Dim;
        __global double* Gb0t = Gb0 + wid_target;
        __global double* Gb1c = Gb1 + wid_context;
        
        int d = 0;
        for (d = 0; d < Dim ; ++d)
        {
            diff += W0t[d]*W1c[d];
        }

        diff += B0t[0] + B1c[0] - log(value);    
        fdiff = value > Xmax ? diff : pow(value/Xmax, Alpha) * diff;
    
        if (isnan(diff) || isinf(diff) || isnan(fdiff) || isinf(fdiff))
            continue;
    
        // weight update
        for (d = 0; d < Dim; ++d)
        {
            grad_w[0] = clamp(fdiff*W1c[d], -100.0, 100.0)*LR;
            grad_w[1] = clamp(fdiff*W0t[d], -100.0, 100.0)*LR;
            upd_w[0] = grad_w[0]/sqrt(Gw0t[d]);
            upd_w[1] = grad_w[1]/sqrt(Gw1c[d]);
            Gw0t[d] += grad_w[0]*grad_w[0];
            Gw1c[d] += grad_w[1]*grad_w[1];
            if( ! isnan(upd_w[0]) && ! isinf(upd_w[0]) )
                W0t[d] -= upd_w[0];
            if( ! isnan(upd_w[1]) && ! isinf(upd_w[1]) )
                W1c[d] -= upd_w[1];
        }
    
        // bias update
        {
            upd_b[0] = fdiff/sqrt(Gb0t[0]);
            upd_b[1] = fdiff/sqrt(Gb1c[0]);
            Gb0t[0] += fdiff*fdiff;
            Gb1c[0] += fdiff*fdiff;
            if( ! isnan(upd_b[0]) && ! isinf(upd_b[0]) )
                B0t[0] -= upd_b[0];
            if( ! isnan(upd_b[1]) && ! isinf(upd_b[1]) )
                B1c[0] -= upd_b[1];
        }
        
        // cost update
        cost += 0.5f*fdiff*fdiff;
        ++count;
    }
    coo_cost[gid] = count > 0 ? cost/count : 0.0;
}
"""

import pyopencl as cl
import numpy as np
import random
import functools
import ctypes

class Glove(object):
    """
    Class for GloVe word embeddings implemented only on python
    """
    def __init__(self, sentences, num_component, min_count=1, max_vocab=0, window_size=15, distance_weighting=True, verbose=False):
        self.cl_context = cl.create_some_context()
        self.cl_queue = cl.CommandQueue(self.cl_context)
        self.cl_program = cl.Program(self.cl_context, cl_source).build()
        self.cl_kernel_update_gradient_immediate = self.cl_program.update_gradient_immediate
        self.cl_kernel_update_gradient_immediate.set_scalar_arg_dtypes([
            np.uint32,
            np.uint32,
            np.float64,
            np.float64,
            np.float64,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            ]
        )
        if verbose:
            local_vars = dict(locals()).copy()
            del local_vars['sentences'], local_vars['self']
            print("[Initialization] parameters = {}".format(local_vars))

        self.num_component = num_component
        self.build_vocabulary(sentences, min_count, max_vocab, verbose=verbose)
        self.count_cooccurrence(sentences, window_size, distance_weighting, verbose=verbose)
        self.initialize_weights()
        
    def build_vocabulary(self, sentences, min_count=1, max_vocab=0, verbose=False):
        if verbose:
            local_vars = dict(locals()).copy()
            del local_vars['sentences'], local_vars['self']
            print("[Building Vocabulary] parameters = {}".format(local_vars))
        word_dict = {}
        for sentence in sentences:
            for word in sentence:
                if type(word) is tuple:
                    word = word[0]
                if word_dict.get(word) is None:
                    word_dict[word] = 0
                word_dict[word] += 1
        self.word_count = [(w,c) for w,c in word_dict.items() if c >= min_count]
        self.word_count.sort(key=functools.cmp_to_key(lambda lhs,rhs : rhs[1] - lhs[1] if rhs[1] != lhs[1] else -1 if lhs[0] < rhs[0] else 1 if lhs[0] > rhs[0] else 0))
        if max_vocab > 0:
            self.word_count = self.word_count[0:max_vocab]
        self.word_to_wid = { wc[0]:i for i, wc in enumerate(self.word_count) }
        self.wid_to_word = { i:wc[0] for i, wc in enumerate(self.word_count) }
        if verbose:
            print("[Building Vocabulary] result = {}".format({'len(words)' : len(self.word_count), 'word[0]' : self.word_count[0], 'word[-1]' : self.word_count[-1]}))
    
    def count_cooccurrence(self, sentences, window_size, distance_weighting, verbose=False):
        if verbose:
            local_vars = dict(locals()).copy()
            del local_vars['sentences'], local_vars['self']
            print("[Counting Cooccurrence] parameters = {}".format(local_vars))

        coo = {}
        for sentence in sentences:
            words = sentence
            for ti, word in enumerate(sentence):
                if self.word_to_wid.get(word) is None:
                    continue
                wid_target = self.word_to_wid[word]
                for ci in range(max(ti-window_size,0),ti): # for words left to target word within window
                    if self.word_to_wid.get(words[ci]) is None:
                        continue
                    wid_context = self.word_to_wid[words[ci]]
                    if wid_target == wid_context:
                        continue
                    key = (wid_target,wid_context)
                    if coo.get(key) is None:
                        coo[key] = 0.0
                    weight = 1.0/(ti-ci) if distance_weighting else 1.0
                    coo[key] += weight
                    rkey = (wid_context,wid_target)
                    if coo.get(rkey) is None:
                        coo[rkey] = 0.0
                    coo[rkey] += weight
        self.coo_records = list(coo.items())
        random.shuffle(self.coo_records)
        if verbose:
            print("[Counting Cooccurrence] result = {}".format({'len(cooccur_list)' : len(self.coo_records), 'max(cooccur_list.count)' : max(self.coo_records, key=lambda x: x[1]), 'min(cooccur_list.count)' : min(self.coo_records, key=lambda x: x[1])}))
    
    def initialize_weights(self):
        self.W = (np.random.rand(2, len(self.word_count), self.num_component).astype(np.float64) - 0.5) / self.num_component
        self.B = (np.random.rand(2, len(self.word_count), 1).astype(np.float64) - 0.5) / self.num_component
        self.Gw = np.ones((2, len(self.word_count), self.num_component), np.float64)
        self.Gb = np.ones((2, len(self.word_count), 1), np.float64)
    
    def fit(self, force_initialize=False, num_iteration=50, num_procs=8192, x_max=100, alpha=0.75, learning_rate=0.05, verbose=False):
        if verbose:
            print("[Training Model] parameters = {}".format(dict(locals())))
        
        if force_initialize:
            initialize_weights()
                
        history = {'loss':[]}
        coo_list = self.coo_records
        N = len(coo_list)
        L = len(self.word_count)
        num_procs = min(num_procs, N//128+1) # to prevent over-parallelization
        Dim = self.num_component
        h_wids_target = np.array([ coo[0][0] for coo in coo_list ], np.uint32)
        h_wids_context = np.array([ coo[0][1] for coo in coo_list ], np.uint32)
        h_values = np.array([ coo[1] for coo in coo_list ], np.float64)
        h_coo_group = np.array([ rank*N//num_procs for rank in range(num_procs + 1) ], np.uint32)
        G = len(h_coo_group) - 1
        h_coo_cost = np.zeros(h_coo_group.size, np.float64)
   
        d_wids_target = cl.Buffer(self.cl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_wids_target)
        d_wids_context = cl.Buffer(self.cl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_wids_context)
        d_values = cl.Buffer(self.cl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_values)
        d_W = cl.Buffer(self.cl_context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.W)
        d_B = cl.Buffer(self.cl_context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.B)
        d_Gw = cl.Buffer(self.cl_context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.Gw)
        d_Gb = cl.Buffer(self.cl_context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.Gb)
        d_coo_group = cl.Buffer(self.cl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_coo_group)
        d_coo_cost = cl.Buffer(self.cl_context, cl.mem_flags.WRITE_ONLY, h_coo_cost.nbytes)

        for iter in range(num_iteration):
            if verbose:
                print("iteration # %d ... " % iter, end="")

            self.cl_kernel_update_gradient_immediate(
                self.cl_queue,
                (G,),
                None,
                L,
                Dim,
                x_max,
                alpha,
                learning_rate,
                d_coo_group,
                d_wids_target,
                d_wids_context,
                d_values,
                d_W,
                d_B,
                d_Gw,
                d_Gb,
                d_coo_cost
            )            
            self.cl_queue.finish()
            cl.enqueue_copy(self.cl_queue, h_coo_cost, d_coo_cost)
            
            history['loss'].append(np.sum(h_coo_cost)/G)
            if verbose:
                print("loss = %f" % history['loss'][-1])
        cl.enqueue_copy(self.cl_queue, self.W, d_W)
        cl.enqueue_copy(self.cl_queue, self.B, d_B)
        cl.enqueue_copy(self.cl_queue, self.Gw, d_Gw)
        cl.enqueue_copy(self.cl_queue, self.Gb, d_Gb)
        self.word_vector = self.W[0] + self.W[1]
        return history
    
    def most_similar(self, word, number=5):
        wid = self.word_to_wid[word]
        word_vec = self.word_vector[wid]
        dst = (np.dot(self.word_vector, word_vec)
               / np.linalg.norm(self.word_vector, axis=1)
               / np.linalg.norm(word_vec))
        word_ids = np.argsort(-dst)
        return [(self.wid_to_word[x], dst[x]) for x in word_ids[:number] if x in self.wid_to_word][1:]