from flask import Flask, request, render_template
from search import Search

app = Flask(__name__)

# Contains the path to the final inverted index stored in partial_indexes_dir from inverted_index.py
INVERTED_INDEX_PATH = ''

search_engine = Search(INVERTED_INDEX_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results', methods=['GET'])
def results():
    query = request.args.get('query', '').strip()
    if not query:
        return render_template('results.html', query=query, results=[], error="Please enter a query.")

    results = search_engine.search(query)
    results = [doc_id for doc_id, _ in results]

    if not results:
        return render_template('results.html', query=query, results=[], error="No results found.")

    return render_template('results.html', query=query, results=results, error=None)

if __name__ == '__main__':
    app.run(debug=True)
