import os
from flask import Flask, render_template, request, redirect, url_for
from openai import OpenAI
from pinecone import Pinecone

# Retrieve API keys from environment variables
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# Initialize lm-studio client
client = OpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = 'arxiv-papers'

# Check if the index exists
if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' does not exist. Please create the index first.")

pc_index = pc.Index(index_name)

app = Flask(__name__)

def get_embedding(text, model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def search_terms(terms):
    embedding = get_embedding(terms)
    query_response = pc_index.query(
        vector=embedding,
        top_k=100,
        include_values=True,
        include_metadata=True
    )
    filenames_summaries = [
        (match['metadata']['file_name'], match['metadata'].get('summary', ''))
        for match in query_response['matches']
    ]
    return filenames_summaries

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        terms = request.form.get('terms')
        show_summaries = 'show_summaries' in request.form
        if terms:
            return redirect(url_for('search', terms=terms, page=1, show_summaries=show_summaries))
    return render_template('index.html')

@app.route('/search')
def search():
    terms = request.args.get('terms')
    page = int(request.args.get('page', 1))
    show_summaries = request.args.get('show_summaries') == 'True'

    filenames_summaries = search_terms(terms)
    page_size = 5
    total_files = len(filenames_summaries)
    total_pages = (total_files + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    filenames_summaries_page = filenames_summaries[start_idx:end_idx]

    return render_template('search.html', terms=terms, page=page, total_pages=total_pages,
                           filenames_summaries=filenames_summaries_page, show_summaries=show_summaries)

if __name__ == '__main__':
    app.run(debug=True, port=5001)

