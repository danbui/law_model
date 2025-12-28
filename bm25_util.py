import math
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pyvi import ViTokenizer

def vi_tokenizer(text):
    return ViTokenizer.tokenize(text).split()

class BM25SparseVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer=None, k1=1.5, b=0.75):
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(tokenizer=self.tokenizer, token_pattern=None)
        self.idf_diag = None
        self.avgdl = 0
        self.n_docs = 0

    def fit(self, raw_documents, y=None):
        """
        Learn vocabulary and IDF from training set.
        """
        # 1. Count term frequencies
        X = self.vectorizer.fit_transform(raw_documents)
        
        # 2. Compute stats
        self.n_docs = X.shape[0]
        self.n_terms = X.shape[1]
        doc_lengths = np.array(X.sum(axis=1)).flatten()
        self.avgdl = doc_lengths.mean()

        # 3. Compute IDF
        # IDF(t) = log( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )
        # equivalent to standard BM25 IDF
        n_t = np.bincount(X.indices, minlength=X.shape[1])
        idf = np.log((self.n_docs - n_t + 0.5) / (n_t + 0.5) + 1)
        
        # Correct potential negative IDFs (stop words) -> usually set to 0 or epsilon
        idf[idf < 0] = 0
        
        self.idf_diag = sparse.diags(idf)
        return self

    def transform(self, raw_documents):
        """
        Transform documents to BM25 sparse vectors.
        """
        # Get Term Frequencies
        X = self.vectorizer.transform(raw_documents)
        
        # Convert to float for division
        X = X.astype(np.float32)
        
        # Get document lengths for this batch
        doc_lengths = np.array(X.sum(axis=1)).flatten()
        
        # Compute BM25 weights
        # TF component: ((k1 + 1) * tf) / (k1 * (1 - b + b * len / avgdl) + tf)
        
        # Denominator part related to doc length
        # shape: (n_docs, 1)
        len_norm = (1 - self.b) + self.b * (doc_lengths / self.avgdl)
        
        # We need to operate on the sparse data directly to save memory/time
        # X is CSR. data is tf.
        
        # Copy to avoid modifying state of X if re-used (though here we don't)
        X_bm25 = X.copy()
        
        # Iterate over rows (docs) to apply length normalization to the TF values
        # This can be vectorized by repeating len_norm for each nonzero element in the row
        # But for Scipy CSR, an easier way is:
        # data[row_start:row_end] / len_norm[row_idx]
        
        # However, to facilitate fast ops, let's use the CSR structure:
        # We need to divide each X[i, j] by (k1 * len_norm[i] + X[i, j])
        # And multiply by (k1 + 1)
        
        for i in range(X.shape[0]):
            start, end = X.indptr[i], X.indptr[i+1]
            if start == end: continue
            
            tf = X.data[start:end]
            norm = len_norm[i]
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + (self.k1 * norm)
            
            X_bm25.data[start:end] = numerator / denominator
            
        # Finally apply IDF
        # X_bm25 * IDF_diag
        X_bm25 = X_bm25 * self.idf_diag
        
        return X_bm25

    def fit_transform(self, raw_documents, y=None):
        return self.fit(raw_documents).transform(raw_documents)
        
    def transform_query(self, query):
        """
        For the query, we arguably just want the terms with weight 1 (or original TF),
        multiplied by their IDF? 
        The Dot Product in Qdrant = Sum( q_i * d_i ).
        Since d_i already includes IDF, if we include IDF in q_i again, it's squared.
        Standard BM25 formula: Sum( IDF * ... ). 
        So q_i should usually just be binary (1) or Term Frequency in Query (QTF).
        Let's stick to binary/QTF.
        """
        # In this specific specific implementation, we return sparse vector 
        # that will be dotted with document vector.
        # Since document vector already contains IDF, we should NOT re-apply IDF 
        # unless we want to emphasize rare query terms even more.
        # Standard: Score = Sum( IDF(qi) * ...doc_part... )
        # Our Doc Vector = IDF(ti) * ...doc_part...
        # So Query Vector should just be QTF (usually 1).
        
        # Just use count vectorizer transform 
        # (This gives TF of query. For short queries, it's usually 1s)
        return self.vectorizer.transform(query)
