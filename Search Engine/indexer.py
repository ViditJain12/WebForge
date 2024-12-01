import os
import json
import pickle
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from math import log
import hashlib

# data_directory contains the path to the directory containing JSON files with a closed set of documents after web crawling - format is {url:, content:, encoding:}
# partial_indexes_dir contains the path to a directory where all partial indexes along with the final index will be stored

class InvertedIndexBuilder:
    def __init__(self, data_directory, partial_indexes_dir, chunk_size=5000):
        self.data_directory = data_directory
        self.partial_indexes_dir = partial_indexes_dir
        self.chunk_size = chunk_size
        self.stemmer = PorterStemmer()
        self.total_documents = 0

        # Tag weights
        self.tag_weights = {
            "title": 3,       # Most important
            "h1": 3,          # High-level heading
            "h2": 2,          # Sub-headings
            "h3": 2,
            "h4": 2,
            "h5": 2,
            "strong": 2,      # Bold or emphasized
            "b": 2,
            "em": 2,          # Italic or emphasized
        }

    def tokenize(self, text):
        return word_tokenize(text.lower())

    def stem(self, word):
        return self.stemmer.stem(word)
    
    def compute_hash(self, content):
        normalized_content = ''.join(content.lower().split())
        return hashlib.md5(normalized_content.encode('utf-8')).hexdigest()  # Computes the MD5 hash of the UTF-8 encoded byte sequence.

    def load_documents(self, directory):
        documents = {}
        seen_hashes = set()

        for subdir, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(subdir, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        doc_id = data.get("url")
                        content = data.get("content", "")
                        doc_hash = self.compute_hash(content)

                        if doc_hash not in seen_hashes:
                            seen_hashes.add(doc_hash)
                            documents[doc_id] = content
                            self.total_documents += 1

        return documents

    def process_document_with_weights(self, content):
        soup = BeautifulSoup(content, 'html.parser')

        term_freq = defaultdict(int)

        # Process weighted tags
        for tag, weight in self.tag_weights.items():
            for element in soup.find_all(tag):
                tokens = [self.stem(token) for token in self.tokenize(element.get_text())]
                for token in tokens:
                    term_freq[token] += weight

        # Process unweighted text (fallback)
        all_text = soup.get_text()
        tokens = [self.stem(token) for token in self.tokenize(all_text)]
        for token in tokens:
            term_freq[token] += 1

        return term_freq

    def build_inverted_index(self, doc_term_freqs):
        inverted_index = defaultdict(list)
        for doc_id, term_freqs in doc_term_freqs.items():
            for term, freq in term_freqs.items():
                inverted_index[term].append({
                    "doc_id": doc_id,
                    "frequency": freq
                })
        return inverted_index

    def save_partial_index(self, index, filename):
        with open(filename, 'wb') as f:
            pickle.dump(index, f)

    def merge_indexes(self, partial_indexes):
        merged_index = defaultdict(list)
        document_frequency = defaultdict(int)

        # Merge partial indexes and calculate document frequency
        for partial_index in partial_indexes:
            for term, postings in partial_index.items():
                merged_index[term].extend(postings)
                document_frequency[term] += len(postings)

        # Calculate TF-IDF
        for term, postings in merged_index.items():
            for posting in postings:
                tf = 1 + log(posting["frequency"])
                idf = log(self.total_documents / document_frequency[term])
                posting["tf_idf"] = tf * idf

        return merged_index

    def get_index_statistics(self, documents, inverted_index, index_file_path):
        num_docs = len(documents)
        num_tokens = len(inverted_index)
        index_size_kb = os.path.getsize(index_file_path) / 1024

        print("Index Statistics:")
        print(f"Number of Documents: {num_docs}")
        print(f"Number of Unique Tokens: {num_tokens}")
        print(f"Total Index Size (KB): {index_size_kb}")

    def run(self):
        documents = self.load_documents(self.data_directory)
        doc_ids = list(documents.keys())

        partial_index_paths = []
        doc_term_freqs = {}
        partial_index_counter = 0

        # Process documents in chunks
        for i in range(0, len(doc_ids), self.chunk_size):
            chunk_doc_ids = doc_ids[i:i + self.chunk_size]

            for doc_id in chunk_doc_ids:
                doc_term_freqs[doc_id] = self.process_document_with_weights(documents[doc_id])

            partial_index = self.build_inverted_index(doc_term_freqs)
            partial_index_path = os.path.join(self.partial_indexes_dir, f'partial_index_{partial_index_counter}.pkl')
            self.save_partial_index(partial_index, partial_index_path)
            partial_index_paths.append(partial_index_path)
            print(f"Saved partial index {partial_index_counter} to {partial_index_path}")

            doc_term_freqs.clear()
            partial_index_counter += 1

        # Merge partial indexes
        partial_indexes = [pickle.load(open(f, 'rb')) for f in partial_index_paths]
        final_index = self.merge_indexes(partial_indexes)

        final_index_path = os.path.join(self.partial_indexes_dir, 'final_index.pkl')
        self.save_partial_index(final_index, final_index_path)
        print(f"Final index saved to {final_index_path}")

        self.get_index_statistics(documents, final_index, final_index_path)


if __name__ == '__main__':
    data_directory = ''
    partial_indexes_dir = ''
    os.makedirs(partial_indexes_dir, exist_ok=True)

    index_builder = InvertedIndexBuilder(data_directory, partial_indexes_dir)
    index_builder.run()
