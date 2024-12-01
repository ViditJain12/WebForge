import pickle
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict

class Search:
    def __init__(self, inverted_index_path):
        self.stemmer = PorterStemmer()
        self.inverted_index = self.load_inverted_index(inverted_index_path)

    @staticmethod
    def tokenize(text):
        """ Tokenize the input text into words """
        return word_tokenize(text.lower())

    def stem(self, word):
        """ Stem a single word using PorterStemmer """
        return self.stemmer.stem(word)

    def load_inverted_index(self, index_path):
        """ Load the inverted index from a file """
        with open(index_path, 'rb') as f:
            return pickle.load(f)

    def boolean_and(self, term1, term2):
        """ Perform a Boolean AND operation between two terms' document sets """
        docs1 = {posting['doc_id'] for posting in self.inverted_index.get(term1, [])}
        docs2 = {posting['doc_id'] for posting in self.inverted_index.get(term2, [])}
        return docs1 & docs2

    def compute_relevance_scores(self, query_terms, candidate_docs):
        """ Compute relevance scores for documents based on TF-IDF, restricted to candidates """
        doc_scores = defaultdict(float)

        for term in query_terms:
            postings = self.inverted_index.get(term, [])
            for posting in postings:
                doc_id = posting["doc_id"]
                if doc_id in candidate_docs:
                    tf_idf = posting.get("tf_idf", 0)
                    doc_scores[doc_id] += tf_idf  # Sum TF-IDF scores for terms in query

        return doc_scores

    def process_query(self, query):
        """ Process the user query: tokenize and stem """
        tokens = [self.stem(token) for token in self.tokenize(query)]
        return tokens

    def filter_candidates(self, query_terms):
        """ Apply Boolean AND logic to filter candidate documents containing all terms """
        if not query_terms:
            return set()

        # Start with the postings list of the first term
        candidate_docs = {posting['doc_id'] for posting in self.inverted_index.get(query_terms[0], [])}

        # Apply Boolean AND for all other terms
        for term in query_terms[1:]:
            term_docs = {posting['doc_id'] for posting in self.inverted_index.get(term, [])}
            candidate_docs &= term_docs  # Intersection

        return candidate_docs

    def search(self, query, top_k=5):
        """ Perform a search using Boolean AND and TF-IDF ranking """
        query_terms = self.process_query(query)
        
        # Apply Boolean AND logic to filter candidates
        candidate_docs = self.filter_candidates(query_terms)

        # Compute relevance scores for the candidates
        doc_scores = self.compute_relevance_scores(query_terms, candidate_docs)
        
        # Sort documents by score in descending order
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return the top-k document IDs and their scores
        return sorted_docs[:top_k]

    def run(self):
        """ Run the search engine in a command-line interface """
        while True:
            # Get user query
            query = input("\nEnter your query (or type 'exit' to exit): ").strip()
            if query.lower() == 'exit':
                print("Goodbye!")
                break
            
            # Perform search
            results = self.search(query)
            
            # Display results - URL + Score (on console)
            if results:
                print("\nTop Results:")
                for i, (doc_id, score) in enumerate(results, start=1):
                    print(f"{i}. {doc_id} - Score: {score:.4f}")
            else:
                print(f"\nNo results found for query ({query}).")

if __name__ == '__main__':
    # Contains the path to the final inverted index stored in partial_indexes_dir from inverted_index.py
    inverted_index_path = ''
    
    search_engine = Search(inverted_index_path)
    search_engine.run()
