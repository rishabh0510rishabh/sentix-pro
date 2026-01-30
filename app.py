from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from typing import Dict, Any, Union

# Ensure consistent results from the language detector
DetectorFactory.seed = 0

app = Flask(__name__)

class SentimentEngine:
    """
    A robust Natural Language Processing engine for Sentiment Analysis.
    
    This class handles the core NLP logic, including polarity scoring,
    subjectivity assessment, and language detection to provide 
    contextual warnings for non-English text.
    """

    @staticmethod
    def analyze(text: str) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the provided text.

        Args:
            text (str): The raw text to be analyzed.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - score (float): The polarity score from -1.0 to 1.0.
                - subjectivity (float): The subjectivity score from 0.0 to 1.0.
                - category (str): Human-readable sentiment category.
                - color (str): CSS-friendly color mapping.
                - rotation (float): Needle rotation degrees for the UI gauge.
                - is_english (bool): Whether the text was detected as English.
        """
        if not text or not text.strip():
            return {"error": "Input text is empty or invalid."}

        # NLP Processing
        blob = TextBlob(text)
        polarity: float = blob.sentiment.polarity
        subjectivity: float = blob.sentiment.subjectivity

        # Language Detection (Warning System)
        is_english: bool = True
        try:
            # We only run detection on meaningful lengths to avoid false positives
            if len(text.strip()) > 10:
                lang = detect(text)
                if lang != 'en':
                    is_english = False
        except Exception:
            # Fallback if detection fails (e.g., numbers, symbols, or service error)
            pass

        # Mapping polarity to Gauge rotation: -1.0 = -90deg, 0.0 = 0deg, 1.0 = 90deg
        rotation_degrees: float = polarity * 90

        # Sentiment Categorization Logic
        if polarity > 0.15:
            category, color = "Positive", "green"
        elif polarity < -0.15:
            category, color = "Negative", "red"
        else:
            category, color = "Neutral", "gray"

        return {
            "score": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "category": category,
            "color": color,
            "rotation": rotation_degrees,
            "is_english": is_english
        }

@app.route('/')
def index() -> str:
    """Renders the main dashboard template."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_endpoint() -> Any:
    """
    AJAX API endpoint for processing sentiment analysis requests.
    
    Expects a JSON payload with a 'text' field.
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request body"}), 400
        
        user_text: str = data['text']
        
        # Security/Performance check: Backend hard-limit
        if len(user_text) > 15000:
            return jsonify({"error": "Text exceeds maximum character limit (15,000)"}), 413

        result = SentimentEngine.analyze(user_text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Start the Flask development server
    app.run(debug=True)