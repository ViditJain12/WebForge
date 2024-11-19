import os
import json
import re
import pickle
from collections import defaultdict
from nltk.stem import PorterStemmer

# data_directory contains the path to the directory containing JSON files with a closed set of documents after web crawling - format is {url:, content:, encoding:}
# partial_indexes_dir contains the path to a directory where all partial indexes along with the final index will be stored

class InvertedIndexBuilder:
    def __init__(self, data_directory, partial_indexes_dir, chunk_size=5000):
        self.data_directory = data_directory
        self.partial_indexes_dir = partial_indexes_dir
        self.chunk_size = chunk_size
        self.stemmer = PorterStemmer()

    # Tokenizer using regex to split on non-alphanumeric characters
    def tokenize(self, text):
        return [token for token in re.split(r'[0-9a-z]+', text.lower()) if token]

    # Stemming function using PorterStemmer
    def stem(self, word):
        return self.stemmer.stem(word)

    # Parse JSON files
    def load_documents(self, directory):
        documents = {}
        for subdir, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(subdir, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        doc_id = data.get("url")  # Using URL as document ID
                        content = data.get("content", "")
                        documents[doc_id] = content
        return documents

    # Tokenize, Stem, and Compute Term Frequency
    def process_document(self, content):
        terms = [self.stem(token) for token in self.tokenize(content)]
        term_freq = defaultdict(int)
        for term in terms:
            term_freq[term] += 1
        return term_freq

    # Build the Inverted Index
    def build_inverted_index(self, doc_term_freqs):
        inverted_index = defaultdict(list)
        for doc_id, term_freqs in doc_term_freqs.items():
            for term, freq in term_freqs.items():
                inverted_index[term].append({
                    "doc_id": doc_id,
                    "frequency": freq
                })
        return inverted_index

    # Save Partial Indexes to Disk
    def save_partial_index(self, index, filename):
        with open(filename, 'wb') as f:
            pickle.dump(index, f)

    # Merge Partial Indexes
    def merge_indexes(self, partial_indexes):
        merged_index = defaultdict(list)
        for partial_index in partial_indexes:
            for term, postings in partial_index.items():
                merged_index[term].extend(postings)
        return merged_index

    # Generate Analytics
    def get_index_statistics(self, documents, inverted_index, index_file_path):
        num_docs = len(documents)
        num_tokens = len(inverted_index)
        index_size_kb = os.path.getsize(index_file_path) / 1024  # Convert to KB

        print("Index Statistics:")
        print(f"Number of Documents: {num_docs}")
        print(f"Number of Unique Tokens: {num_tokens}")
        print(f"Total Index Size (KB): {index_size_kb}")

    # Main function to execute the milestone tasks
    def run(self):
        documents = self.load_documents(self.data_directory)
        doc_ids = list(documents.keys())

        partial_index_paths = []
        doc_term_freqs = {}
        partial_index_counter = 0

        # Process documents in chunks
        for i in range(0, len(doc_ids), self.chunk_size):
            chunk_doc_ids = doc_ids[i:i + self.chunk_size]

            # Process the current chunk
            for doc_id in chunk_doc_ids:
                doc_term_freqs[doc_id] = self.process_document(documents[doc_id])

            # Build and save partial index for this chunk
            partial_index = self.build_inverted_index(doc_term_freqs)
            partial_index_path = os.path.join(self.partial_indexes_dir, f'partial_index_{partial_index_counter}.pkl')
            self.save_partial_index(partial_index, partial_index_path)
            partial_index_paths.append(partial_index_path)
            print(f"Saved partial index {partial_index_counter} to {partial_index_path}")

            # Clear memory and prepare for the next chunk
            doc_term_freqs.clear()
            partial_index_counter += 1

        # Merge all partial indexes
        partial_indexes = [pickle.load(open(f, 'rb')) for f in partial_index_paths]
        final_index = self.merge_indexes(partial_indexes)

        # Save the final merged index
        final_index_path = os.path.join(self.partial_indexes_dir, 'final_index.pkl')
        self.save_partial_index(final_index, final_index_path)
        print(f"Final index saved to {final_index_path}")

        # Get statistics about the index
        self.get_index_statistics(documents, final_index, final_index_path)


if __name__ == '__main__':
    data_directory = ''
    partial_indexes_dir = ''
    os.makedirs(partial_indexes_dir, exist_ok=True)

    index_builder = InvertedIndexBuilder(data_directory, partial_indexes_dir)
    index_builder.run()
