import unittest
import json
from app import SentimentEngine

class TestSentimentEngine(unittest.TestCase):
    """
    Test suite for the Sentix Sentiment Engine.
    Covers polarity classification, language warnings, and edge cases.
    """

    def setUp(self):
        """Prepare fresh engine instances if needed."""
        self.engine = SentimentEngine()

    def test_positive_classification(self):
        """Should identify clearly positive text."""
        result = self.engine.analyze("I absolutely love this project! It's brilliant.")
        self.assertEqual(result['category'], "Positive")
        self.assertGreater(result['score'], 0.15)
        self.assertEqual(result['color'], "green")

    def test_negative_classification(self):
        """Should identify clearly negative text."""
        result = self.engine.analyze("This is a horrible experience. I'm very disappointed.")
        self.assertEqual(result['category'], "Negative")
        self.assertLess(result['score'], -0.15)
        self.assertEqual(result['color'], "red")

    def test_neutral_classification(self):
        """Should identify factual or neutral text."""
        result = self.engine.analyze("The table is sitting in the middle of the room.")
        self.assertEqual(result['category'], "Neutral")
        self.assertTrue(-0.15 <= result['score'] <= 0.15)

    def test_language_detection(self):
        """Should flag non-English text."""
        # Spanish text
        result = self.engine.analyze("Hola, este es un texto en español para probar el sistema.")
        self.assertFalse(result['is_english'])

    def test_empty_input(self):
        """Should handle empty strings gracefully."""
        result = self.engine.analyze("   ")
        self.assertIn("error", result)

if __name__ == '__main__':
    print("Running Sentix Engine Verification Suite...")
    unittest.main()