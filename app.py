import os
from flask import Flask, render_template, request, redirect, url_for, render_template_string, abort
from openai import OpenAI
from pinecone import Pinecone
from flask import send_from_directory
import markdown
import requests
from datetime import datetime
import re
from flask import Flask

app = Flask(__name__)

@app.template_filter('safe_name')
def safe_name_filter(s):
    return re.sub(r'\W+', '_', s)


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

paper_dir = "/ai/fabric/output/arvix_papers/"



@app.errorhandler(400)
def bad_request(e):
    return render_template('400.html'), 400

@app.errorhandler(401)
def unauthorized(e):
    return render_template('401.html'), 401

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500




@app.route('/view_paper/<title>')
def view_paper(title):
    # Generate a safe name from the title
    safe_name = re.sub(r'\W+', '_', title) + '.md'

    # Construct the URL of the markdown file
    url = f"https://raw.githubusercontent.com/osok/arXiv_papers/main/{safe_name}"

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        abort(404, "File not found")

    # Get the content of the file
    content = response.text

    # Convert the markdown to HTML
    html_content = markdown.markdown(content)

    # Render the HTML
    return render_template_string(html_content)


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
        {
            'file_name': match['metadata']['file_name'],
            'summary': match['metadata'].get('summary', ''),
            'title': match['metadata'].get('title', ''),
            'short_url': match['metadata'].get('short_url', ''),
            'views': match['metadata'].get('views', ''),
            'likes': match['metadata'].get('likes', ''),
            'video_date': datetime.strptime(match['metadata'].get('video_date', ''), '%Y-%m-%dT%H:%M:%SZ') if match['metadata'].get('video_date') else ''  # Convert video_date to datetime
        }
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
    page_size = 20
    total_files = len(filenames_summaries)
    total_pages = (total_files + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    filenames_summaries_page = filenames_summaries[start_idx:end_idx]

    return render_template('search.html', terms=terms, page=page, total_pages=total_pages,
                           filenames_summaries=filenames_summaries_page, show_summaries=show_summaries)

if __name__ == '__main__':
    app.run(debug=True, port=5001)

