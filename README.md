# MoodMatch-An-AI-Powered-Book-Recommender-
MoodMatch is an intelligent book recommendation system that leverages natural language understanding to suggest books based on user-described moods, themes, or emotions.
📦 Features
🔍 Semantic Search using Sentence Transformers (MiniLM)

📚 Personalized book suggestions with emotion-based filtering

🎨 Clean UI built with HTML + Bootstrap

⚙️ Powered by Flask, LangChain, ChromaDB, and HuggingFace Embeddings

📂 Folder Structure
php
Copy
Edit
MoodMatch/
│
├── app.py                   # Flask backend and recommendation logic
├── templates/
│   └── index.html           # Frontend UI
├── static/
│   ├── style.css            # Custom styling (optional)
│   └── favicon.ico          # Browser icon (optional)
├── tagged_description.txt   # Raw book data (tagged)
├── books_with_emotions.csv  # Dataset with emotion + genre metadata
├── requirements.txt         # Python libraries
└── README.md                # Project documentation
