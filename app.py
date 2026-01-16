from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from typing import Dict, Any

# Ensure consistent language detection results
DetectorFactory.seed = 0

app = Flask(__name__)

class SentimentEngine:
    """
    Core engine for processing Natural Language data.
    Separates NLP logic from web routing for better maintainability.
    """

    @staticmethod
    def analyze(text: str) -> Dict[str, Any]:
        """
        Performs sentiment analysis and language detection on the input text.
        
        Args:
            text (str): The raw text string to analyze.
            
        Returns:
            dict: A collection of scores and metadata for the UI.
        """
        if not text.strip():
            return {"error": "No text provided"}

        # Perform Sentiment Analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Language Detection Logic 
        is_english = True
        try:
            if len(text.strip()) > 3: # Need a minimum length for accurate detection
                lang = detect(text)
                if lang != 'en':
                    is_english = False
        except Exception:
            # Fallback if detection fails (e.g., gibberish or numbers)
            pass

        # Speedometer Logic: -1.0 is -90deg, 0.0 is 0deg, 1.0 is 90deg
        rotation = polarity * 90

        # Classification Mapping
        if polarity > 0.15:
            category, color = "Positive", "green"
        elif polarity < -0.15:
            category, color = "Negative", "red"
        else:
            category, color = "Neutral", "gray"

        return {
            "score": round(polarity, 2),
            "subjectivity": round(subjectivity, 2),
            "category": category,
            "color": color,
            "rotation": rotation,
            "is_english": is_english
        }

@app.route('/')
def index():
    """Renders the dashboard UI."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    """API endpoint that receives text and returns sentiment JSON."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid request"}), 400
        
    result = SentimentEngine.analyze(data['text'])
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)