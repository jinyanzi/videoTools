"""
(C) Mathieu Blondel - 2010
License: BSD 3 clause

Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in

Finding scientifc topics (Griffiths and Steyvers)
"""

import numpy as np
import math
import scipy as sp
import scipy.misc 
from scipy.special import gammaln
import math
import cv2
from itertools import product

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx

def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)

class LdaSampler(object):

    def __init__(self, n_topics, alpha=0.1, beta=0.1):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta

    def _initialize(self, matrix):
        n_docs, vocab_size = matrix.shape

        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics))
        # number of times topic z and word w co-occur
        self.nzw = np.zeros((self.n_topics, vocab_size))
        self.nm = np.zeros(n_docs)
        self.nz = np.zeros(self.n_topics)
        self.topics = {}

        for m in xrange(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                self.nmz[m,z] += 1
                self.nm[m] += 1
                self.nzw[z,w] += 1
                self.nz[z] += 1
                self.topics[(m,i)] = z

    def _conditional_distribution(self, m, w):
        """
        Conditional distribution (vector of size n_topics).
        """
        vocab_size = self.nzw.shape[1]
        left = (self.nzw[:,w] + self.beta) / \
               (self.nz + self.beta * vocab_size)
        right = (self.nmz[m,:] + self.alpha) / \
                (self.nm[m] + self.alpha * self.n_topics)
        p_z = left * right
        # normalize to obtain probabilities
        p_z /= np.sum(p_z)
        return p_z

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.nzw.shape[1]
        n_docs = self.nmz.shape[0]
        lik = 0

        for z in xrange(self.n_topics):
            lik += log_multi_beta(self.nzw[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m in xrange(n_docs):
            lik += log_multi_beta(self.nmz[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        V = self.nzw.shape[1]
        num = self.nzw + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def run(self, matrix, maxiter=30):
        """
        Run the Gibbs sampler.
        """
        n_docs, vocab_size = matrix.shape
        self._initialize(matrix)

        for it in xrange(maxiter):
            for m in xrange(n_docs):
                for i, w in enumerate(word_indices(matrix[m, :])):
                    z = self.topics[(m,i)]
                    self.nmz[m,z] -= 1
                    self.nm[m] -= 1
                    self.nzw[z,w] -= 1
                    self.nz[z] -= 1

                    p_z = self._conditional_distribution(m, w)
                    z = sample_index(p_z)

                    self.nmz[m,z] += 1
                    self.nm[m] += 1
                    self.nzw[z,w] += 1
                    self.nz[z] += 1
                    self.topics[(m,i)] = z

            # FIXME: burn-in and lag!
            yield self.phi()

if __name__ == "__main__":
    import os
    import shutil

    N_TOPICS = 5
    DOCUMENT_LENGTH = 100
    FOLDER = "topicimg"

    def vertical_topic(width, topic_index, document_length):
        """
        Generate a topic whose words form a vertical bar.
        """
        m = np.zeros((width, width))
        m[:, topic_index] = int(document_length / width)
        return m.flatten()

    def horizontal_topic(width, topic_index, document_length):
        """
        Generate a topic whose words form a horizontal bar.
        """
        m = np.zeros((width, width))
        m[topic_index, :] = int(document_length / width)
        return m.flatten()

    def save_document_image(filename, doc, zoom=2):
        """
        Save document as an image.

        doc must be a square matrix
        """
        height, width = doc.shape
        zoom = np.ones((width*zoom, width*zoom))
        # imsave scales pixels between 0 and 255 automatically
        sp.misc.imsave(filename, np.kron(doc, zoom))

    def gen_word_distribution(n_topics, document_length):
        """
        Generate a word distribution for each of the n_topics.
        """
        width = n_topics / 2
        vocab_size = width ** 2
        m = np.zeros((n_topics, vocab_size))

        for k in range(width):
            m[k,:] = vertical_topic(width, k, document_length)

        for k in range(width):
            m[k+width,:] = horizontal_topic(width, k, document_length)

        m /= m.sum(axis=1)[:, np.newaxis] # turn counts into probabilities

        return m

    def gen_document(word_dist, n_topics, vocab_size, length=DOCUMENT_LENGTH, alpha=0.1):
        """
        Generate a document:
            1) Sample topic proportions from the Dirichlet distribution.
            2) Sample a topic index from the Multinomial with the topic
               proportions from 1).
            3) Sample a word from the Multinomial corresponding to the topic
               index from 2).
            4) Go to 2) if need another word.
        """
        theta = np.random.mtrand.dirichlet([alpha] * n_topics)
        v = np.zeros(vocab_size)
        for n in range(length):
            z = sample_index(theta)
            w = sample_index(word_dist[z,:])
            v[w] += 1
        return v

    def gen_documents(word_dist, n_topics, vocab_size, n=500):
        """
        Generate a document-term matrix.
        """
        m = np.zeros((n, vocab_size))
        for i in xrange(n):
            m[i, :] = gen_document(word_dist, n_topics, vocab_size)
        return m

    def vis_of_document(img, win_idx, doc, step_size):
        cell_h, cell_w, direction = doc.shape
        im_toshow = np.copy(img)

        hsv = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        angle = np.array([0, 45, 90, 135])

        print np.amax(doc)
        max_direction = cv2.normalize(np.amax(doc, axis=2), None, 0, 255, cv2.NORM_MINMAX)
        max_direction[max_direction>0] = 255
        max_direction_idx = np.argmax(doc, axis=2)
        hsv[max_direction_idx==0, 0] = angle[0]
        hsv[max_direction_idx==1, 0] = angle[1]
        hsv[max_direction_idx==2, 0] = angle[2]
        hsv[max_direction_idx==3, 0] = angle[3]
        hsv[...,1] = 255
        hsv[...,2] = cv2.normalize(max_direction, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        for i, j in product(xrange(cell_h), xrange(cell_w)):
            h_idx = i*step_size
            w_idx = j*step_size

            if np.any(bgr[i, j]):
                # print (i, j , bgr[i, j], hsv[i, j])
                im_toshow[h_idx:h_idx+step_size, w_idx:w_idx+step_size] = bgr[i, j]

        cv2.addWeighted(im_toshow, 0.5, img, 0.5, 0, im_toshow)
        win_name = "topic "+str(win_idx)
        x = img.shape[1]*(win_idx%5)
        y = img.shape[0]*(win_idx/5)
        cv2.imshow(win_name, im_toshow)
        cv2.moveWindow(win_name, x, y)

    def gen_of_documents(video_path, n_docs = 100, doc_interval = 10, out_file=None, step_size = 10, static_dist = 10):
        """ Generate optical flow documents from video """
        """ Generate a list of documents from a video"""
        print "Open video ", video_path
        cap = cv2.VideoCapture(video_path)

        ret, frame1 = cap.read()
        n_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        last_frame_idx = doc_interval * n_docs
        
        if n_frame < last_frame_idx or n_docs <= 0:
            last_frame_idx = n_frame
            n_docs = n_frame/doc_interval

        print "last frame ", last_frame_idx, " total ", n_frame
        height, width, channels = frame1.shape
        """ grid coordinate of each cell """
    	x = np.arange(step_size/2, width, step = step_size)
	y = np.arange(step_size/2, height, step = step_size)
        nw_cell = len(x)
        nh_cell = len(y)
        vocab_size = nw_cell*nh_cell*4
 
        docs = np.zeros((n_docs, vocab_size), dtype = np.int32)
    
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

        i = 1
        while(i <= last_frame_idx):
            ret, frame = cap.read()
            if not ret:
                break

            for j in xrange(frames_per_doc):
                if to_read:
                    ret, frame = cap.read()
                else
                    break

                if not ret:
                    to_read = False
            
            print "frame ", i
            current = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs,current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            """ initialization of a document """
            if i % doc_interval == 1:
                d = np.zeros((nh_cell, nw_cell, 4))
                accumulation = np.zeros((nh_cell, nw_cell, 2), dtype=np.float64)
           
            accumulation += flow[step_size/2::step_size, step_size/2::step_size]

            """ Last frame of the interval, generate a document
            compute document every doc_interval frames
            4 directions 
            """
            if i % doc_interval == 0:
                mag, ang = cv2.cartToPolar(accumulation[...,0], accumulation[...,1])
                pi_div = np.pi/4
                ne = np.bitwise_and(mag>static_dist , np.bitwise_or(np.bitwise_and(ang>=0, ang<pi_div), np.bitwise_and(ang>=pi_div*7, ang<pi_div*8)))
                if np.any(ne):
                    d[ne ,0] = 1
                se = np.bitwise_and(mag>static_dist , np.bitwise_and(ang>=pi_div, ang<pi_div*3))
                if np.any(se):
                    d[se,1] = 1
                nw = np.bitwise_and(mag>static_dist , np.bitwise_and(ang>=pi_div*3, ang<pi_div*5))
                if np.any(nw):
                    d[nw,2] = 1
                sw = np.bitwise_and(mag>static_dist , np.bitwise_and(ang>=pi_div*5, ang<pi_div*7))
                if np.any(sw):
                    d[sw,3] = 1
                docs[i/doc_interval-1, :] = d.flatten()
                print "doc %d: %d ne, %d se, %d nw, %d sw, %d total"% ((i/doc_interval-1), np.sum(d[...,0]), np.sum(d[...,1]), np.sum(d[...,2]), np.sum(d[...,3]), np.sum(d))
            i += 1

        print np.sum(docs), docs.shape
        if out_file:
            print "save to ", out_file
            np.savetxt(out_file, docs)
        return docs, nh_cell, nw_cell


    if os.path.exists(FOLDER):
        shutil.rmtree(FOLDER)
    os.mkdir(FOLDER)

    step_size = 10
    # matrix, nh_cell, nw_cell = gen_of_documents('/home/jenny/Desktop/vaticAnnotation/trimmedVideos/193402_Main_St_(US_51_Bus)_and_Empire_St_(IL_9)_in_Bloomington_20141023_11am.mp4', n_docs = 100, out_file = '193402_vocab_100.txt', step_size = step_size)
    img = cv2.imread('/home/jenny/Desktop/vaticAnnotation/screenshots/193402_Main_St_(US_51_Bus)_and_Empire_St_(IL_9)_in_Bloomington_20141023_11am.jpg')
    height, width, ch = img.shape
    nw_cell = len(np.arange(step_size/2, width, step = step_size))
    nh_cell = len(np.arange(step_size/2, height, step = step_size))

    matrix = np.loadtxt('193402_vocab_100.txt', delimiter=' ', dtype=np.uint32)
    print matrix.shape
    
    sampler = LdaSampler(N_TOPICS)

    for it, phi in enumerate(sampler.run(matrix, maxiter=10)):
        print "Iteration", it
        print "Likelihood", sampler.loglikelihood()

        if it % 10 == 9:
            for z in range(N_TOPICS):
                vis_of_document(img, z, phi[z,:].reshape(nh_cell, nw_cell, 4), step_size = 10)
            cv2.waitKey(0)
                #save_document_image("topicimg/topic%d-%d.png" % (it,z),
                                    #phi[z,:].reshape(height, width, 4)[...,1])

    # width = N_TOPICS / 2
    # vocab_size = width ** 2
    # word_dist = gen_word_distribution(N_TOPICS, DOCUMENT_LENGTH)
    # matrix = gen_documents(word_dist, N_TOPICS, vocab_size)
    # sampler = LdaSampler(N_TOPICS)

    # for it, phi in enumerate(sampler.run(matrix)):
    #     print "Iteration", it
    #     print "Likelihood", sampler.loglikelihood()

    #     if it % 5 == 0:
    #         for z in range(N_TOPICS):
    #             save_document_image("topicimg/topic%d-%d.png" % (it,z),
    #                                 phi[z,:].reshape(width,-1))


