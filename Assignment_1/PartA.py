import re
import sys
import os
from collections import defaultdict

FILEMAXCOUNT = 1

class FileToken:
    # Time Complexity: O(N), where n is the total number of characters in the file.
    # Each line is processed once and split into words.
    # Searching and inserting into a list are O(1) on average per word.    
    def tokenize(self, filepath: str) -> list:
        token_lst = []

        try:
            file = open(filepath, 'r', encoding='utf-8')
            for line in file:
                line_split = re.split(r'[^a-zA-Z0-9]+', line.strip())
                for word in line_split:
                    if word != "" and word.isascii():
                        token_lst.append(word.lower()) 
        except FileNotFoundError:
            print(f"The file '{filepath}' was not found") 
            sys.exit()     
        except Exception as e:
            print(f"An error occured while opening the file {e}")  
            sys.exit()

        return token_lst

    # Time Complexity: O(N), where m is the number of tokens (words) in the token list.
    # Each token is processed once to update the frequency count.
    def computeWordFrequencies(self, tokens: list) -> dict:
        tokens_map = defaultdict(str)
        for token in tokens:
            if token in tokens_map:
                tokens_map[token] += 1
            else:
                tokens_map[token] = 1
        return tokens_map
    
    # Time Complexity: O(m log m), where m is the number of unique tokens (words).
    # Sorting the dictionary by values takes O(m log m).
    # Printing takes O(m).
    def print(self, tokens_map: dict) -> None:
        tokens_map = dict(sorted(tokens_map.items(), key=lambda word: word[1], reverse=True))
        for token, frequency in tokens_map.items():
            print(f"{token} = {frequency}")

# Time Complexity: O(N), where d is the number of directories/files in the current directory and its subdirectories.
# os.walk() traverses each directory once.
def get_file_path(index: int) -> str:
    # Index is used to allow for fetching filename or filepath

    # Checking if only index number of filename / filepath have been provided
    if len(sys.argv) != FILEMAXCOUNT + 1: 
        print(f"Provide only {FILEMAXCOUNT} File(s) or Filepath(s)")
        sys.exit()

    file = sys.argv[index]

    # Checking whether the input file is either an absolute path (a whole path is given starting from the root directory) or contains a directory component
    if os.path.isabs(file) or os.path.dirname(file):
        file_path = sys.argv[index]
    else:
        current_directory = os.getcwd()
        file_path = None

        # Getting the whole path of the file with all of it's subdirectories
        for root, dirs, files in os.walk(current_directory):
            if file in files:
                file_path = os.path.join(root, file)

        # Print error message if file path can not be found
        if not file_path:
            print(f"File '{file}' not found in any subdirectory.")
            sys.exit()

    return file_path

if __name__ == "__main__":
    file_path = get_file_path(1)
    filetoken = FileToken()
    tokens = filetoken.tokenize(file_path)
    tokens_map = filetoken.computeWordFrequencies(tokens)
    filetoken.print(tokens_map)
