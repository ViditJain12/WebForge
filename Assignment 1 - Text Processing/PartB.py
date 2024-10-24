import PartA as A

A.FILEMAXCOUNT = 2

# Time Complexity: O(N + M), where n is the number of tokens in filepath_a and m is the number of tokens in filepath_b.
# Tokenizing and computing frequencies for each file are O(N) and O(M) respectively.
def compare(filepath_a: str, filepath_b: str) -> int:
    count = 0

    filetoken_a = A.FileToken()
    tokens_a = filetoken_a.tokenize(filepath_a)
    tokens_map_a = filetoken_a.computeWordFrequencies(tokens_a)

    filetoken_b = A.FileToken()
    tokens_b = filetoken_b.tokenize(filepath_b)
    tokens_map_b = filetoken_b.computeWordFrequencies(tokens_b)

    for token in tokens_map_a:
        if token in tokens_map_b:
            count += 1
    
    return count

if __name__ == "__main__":
    filepath_a = A.get_file_path(1)
    filepath_b = A.get_file_path(2)
    common_ct = compare(filepath_a, filepath_b)
    print(common_ct)
