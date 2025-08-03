
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

app = Flask(__name__)

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].fillna("static/img/fallback.jpg") + "&fife=w800"

# Fix: Use keyword arguments for CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50
)
documents = text_splitter.split_documents(
    TextLoader("tagged_description.txt", encoding="utf-8").load()
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding=embedding_model)

@app.route("/")
def home():
    categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
    tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    return render_template("index.html", categories=categories, tones=tones)

@app.route("/recommend", methods=["POST"])
def recommend():
    query = request.form.get("query", "")
    category = request.form.get("category", "All")
    tone = request.form.get("tone", "All")

    recs = db_books.similarity_search(query, k=50)
    ids = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    results = books[books["isbn13"].isin(ids)].copy()

    if category != "All":
        results = results[results["simple_categories"] == category]

    tone_map = {
        "Happy": "joy", "Surprising": "surprise",
        "Angry": "anger", "Suspenseful": "fear",
        "Sad": "sadness"
    }
    if tone in tone_map:
        results = results.sort_values(by=tone_map[tone], ascending=False)

    final = []
    for _, row in results.head(16).iterrows():
        desc = " ".join(str(row["description"]).split()[:30]) + "..."
        authors = ", ".join(str(row["authors"]).split(";"))
        final.append({
            "image": row["large_thumbnail"],
            "title": row["title"],
            "authors": authors,
            "desc": desc
        })

    return jsonify(final)

if __name__ == "__main__":
    app.run(debug=True)
