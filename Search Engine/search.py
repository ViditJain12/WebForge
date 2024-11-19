import pickle
import re
from nltk.stem import PorterStemmer


class Search:
    def __init__(self, inverted_index_path):
        self.stemmer = PorterStemmer()
        self.inverted_index = self.load_inverted_index(inverted_index_path)

    @staticmethod
    def tokenize(text):
        return [token for token in re.split(r'\W+', text.lower()) if token]

    def stem(self, word):
        return self.stemmer.stem(word)

    def load_inverted_index(self, index_path):
        with open(index_path, 'rb') as f:
            return pickle.load(f)

    def boolean_and(self, term1, term2):
        docs1 = {posting['doc_id'] for posting in self.inverted_index.get(term1, [])}
        docs2 = {posting['doc_id'] for posting in self.inverted_index.get(term2, [])}
        return docs1 & docs2

    def process_query(self, query):
        # Tokenize and stem the query
        tokens = [self.stem(token) for token in self.tokenize(query)]
        
        if not tokens:
            return set()  # No results for empty queries
        
        # If only one term, return its postings directly
        if len(tokens) == 1:
            term = tokens[0]
            return {posting['doc_id'] for posting in self.inverted_index.get(term, [])}
        
        # Handle multi-word queries by processing term pairs
        term_pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        
        # Perform AND operation for the first pair
        results = self.boolean_and(*term_pairs[0])
        
        # Intersect with results for subsequent pairs
        for term1, term2 in term_pairs[1:]:
            results &= self.boolean_and(term1, term2)
        
        # If the number of terms is odd, intersect with the final leftover term
        if len(tokens) % 2 != 0:
            last_term = tokens[-1]
            last_term_docs = {posting['doc_id'] for posting in self.inverted_index.get(last_term, [])}
            results &= last_term_docs
        
        return results

    def search(self, query, top_k=5):
        matching_doc_ids = self.process_query(query)
        matching_urls = [doc_id for doc_id in matching_doc_ids]
        return matching_urls[:top_k]

    def run(self):
        print("Simple Search Engine")
        print("Type 'exit' to quit.")
        
        while True:
            # Get user query
            query = input("\nEnter your query: ").strip()
            if query.lower() == 'exit':
                print("Goodbye!")
                break
            
            # Perform search
            results = self.search(query)
            
            # NOTE: THIS IS THE CONSOLE INTERFACE
            # TO RUN THE WEB INTERFACE, RUN Python3 app.py IN THE CONSOLE

            # Display results
            if results:
                print("\nTop Results:")
                for i, url in enumerate(results, start=1):
                    print(f"{i}. {url}")
            else:
                print("\nNo results found.")


if __name__ == '__main__':
    # Contains the path to the final inverted index stored in partial_indexes_dir from inverted_index.py
    inverted_index_path = ''
    search_engine = Search(inverted_index_path)
    search_engine.run()
