from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class TestCases:
    @staticmethod
    def get_test_cases() -> List[Dict]:
        return [
            {
                'name': 'Basic Question',
                'question': 'What is the main topic of this document?',
                'expected_type': 'summary',
                'description': 'Should summarize the main topic from the document'
            }
        ]
