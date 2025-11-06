import re
import logging
import sqlite3
import uuid
from datetime import datetime
import os
import genai
from transformers import pipeline

your_api_key = 'YOUR_API_KEY_HERE '  # Replace with your actual API key
genai.configure(api_key=your_api_key)

class GuardrailDB:
    def __init__(self, db_path="backend/app/databases/omega.db"):
        self.db_path = db_path

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_table()


    def _create_table(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS guardrails (
                        guardrail_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        tenant_id TEXT,
                        category TEXT,
                        purpose TEXT,
                        filtering_type TEXT,
                        for_prompts_or_responses TEXT,
                        strength_of_filter TEXT,
                        created_at TEXT,
                        example TEXT,
                        response_message TEXT
                    )
                ''')
                conn.commit()
        except Exception as e:
            logging.error(f"Error creating guardrail table: {str(e)}")

    def store_guardrail_hit(self, user_id, tenant_id, guardrail_info):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                guardrail_id = str(uuid.uuid4())
                created_at = datetime.now().isoformat()
                
                cursor.execute('''
                    INSERT INTO guardrails (
                        guardrail_id,
                        user_id,
                        tenant_id,
                        category,
                        purpose,
                        filtering_type,
                        for_prompts_or_responses,
                        strength_of_filter,
                        created_at,
                        example,
                        response_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    guardrail_id,
                    user_id,
                    tenant_id,
                    guardrail_info['category'],
                    guardrail_info['purpose'],
                    guardrail_info['filtering_type'],
                    guardrail_info['for_prompts_or_responses'],
                    guardrail_info['strength_of_filter'],
                    created_at,
                    guardrail_info['example'],
                    guardrail_info['response_message']
                ))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Error storing guardrail hit: {str(e)}")
            return False

class ContentAnalyzer:
    def __init__(self):
        self._hate_classifier = None
        self._toxicity_classifier = None
        self._insult_classifier = None
        self._nsfw_classifier = None
        self._misconduct_classifier = None

    @property
    def hate_classifier(self):
        if self._hate_classifier is None:
            self._hate_classifier = pipeline("text-classification", 
                model="facebook/roberta-hate-speech-dynabench-r4-target")
        return self._hate_classifier

    @property
    def toxicity_classifier(self):
        if self._toxicity_classifier is None:
            self._toxicity_classifier = pipeline("text-classification", 
                model="unitary/toxic-bert")
        return self._toxicity_classifier

    @property
    def insult_classifier(self):
        if self._insult_classifier is None:
            self._insult_classifier = pipeline("text-classification", 
                model="martin-ha/toxic-comment-model")
        return self._insult_classifier

    @property
    def nsfw_classifier(self):
        if self._nsfw_classifier is None:
            self._nsfw_classifier = pipeline("text-classification", 
                model="michellejieli/NSFW_text_classifier")
        return self._nsfw_classifier

    @property
    def misconduct_classifier(self):
        if self._misconduct_classifier is None:
            self._misconduct_classifier = pipeline("text-classification", 
                model="unitary/unbiased-toxic-roberta")
        return self._misconduct_classifier

class Guardrails:
    def __init__(self):
        self.content_analyzer = ContentAnalyzer()
        self.db = GuardrailDB()

        self.filters = {
            "Content Filters": {
                "Hate Filter": self.check_hate_speech,
                "Insult Filter": self.check_insult,
                "Sexual Filter": self.check_sexual_content,
                "Violence Filter": self.check_violence,
                "Misconduct Filter": self.check_misconduct,
            },
            "Denied Topics": {
                "Politics": self.check_politics,
                "Religion": self.check_religion,
                "Legal Advice": self.check_legal_advice,
            },
            "Word Filters": {
                "Profanity": self.check_profanity,
            },
            "Sensitive Information": {
                "PII": self.check_pii,
            },
        }

    def check_hate_speech(self, text, user_id, tenant_id):
        result = self.content_analyzer.hate_classifier(text)[0]
        if result['label'] == 'HATE_SPEECH' and result['score'] > 0.7:
            guardrail_info = {
                'category': 'Content Filter',
                'purpose': 'Prevent hate speech',
                'filtering_type': 'Content Filtering',
                'for_prompts_or_responses': 'Both',
                'strength_of_filter': 'High',
                'example': text[:100],
                'response_message': 'Hate speech detected'
            }
            self.db.store_guardrail_hit(user_id, tenant_id, guardrail_info)
            return False
        return True

    def check_insult(self, text, user_id, tenant_id):
        result = self.content_analyzer.insult_classifier(text)[0]
        if result['label'] == 'toxic' and result['score'] > 0.7:
            guardrail_info = {
                'category': 'Content Filter',
                'purpose': 'Prevent insults',
                'filtering_type': 'Content Filtering',
                'for_prompts_or_responses': 'Both',
                'strength_of_filter': 'High',
                'example': text[:100],
                'response_message': 'Insult detected'
            }
            self.db.store_guardrail_hit(user_id, tenant_id, guardrail_info)
            return False
        return True

    def check_sexual_content(self, text, user_id, tenant_id):
        result = self.content_analyzer.nsfw_classifier(text)[0]
        if result['label'] == 'NSFW' and result['score'] > 0.7:
            guardrail_info = {
                'category': 'Content Filter',
                'purpose': 'Prevent NSFW content',
                'filtering_type': 'Content Filtering',
                'for_prompts_or_responses': 'Both',
                'strength_of_filter': 'High',
                'example': text[:100],
                'response_message': 'NSFW content detected'
            }
            self.db.store_guardrail_hit(user_id, tenant_id, guardrail_info)
            return False
        return True

    def check_violence(self, text, user_id, tenant_id):
        result = self.content_analyzer.toxicity_classifier(text)[0]
        if result['score'] > 0.8:
            guardrail_info = {
                'category': 'Content Filter',
                'purpose': 'Prevent violent content',
                'filtering_type': 'Content Filtering',
                'for_prompts_or_responses': 'Both',
                'strength_of_filter': 'High',
                'example': text[:100],
                'response_message': 'Violent content detected'
            }
            self.db.store_guardrail_hit(user_id, tenant_id, guardrail_info)
            return False
        return True

    def check_misconduct(self, text, user_id, tenant_id):
        result = self.content_analyzer.misconduct_classifier(text)[0]
        if result['label'] == 'toxic' and result['score'] > 0.7:
            guardrail_info = {
                'category': 'Content Filter',
                'purpose': 'Prevent misconduct',
                'filtering_type': 'Content Filtering',
                'for_prompts_or_responses': 'Both',
                'strength_of_filter': 'High',
                'example': text[:100],
                'response_message': 'Misconduct detected'
            }
            self.db.store_guardrail_hit(user_id, tenant_id, guardrail_info)
            return False
        return True

    def check_politics(self, text, user_id, tenant_id):
        politics_keywords = ['politics', 'election', 'voting', 'democrat', 'republican', 'congress', 'senate']
        if any(keyword in text.lower() for keyword in politics_keywords):
            guardrail_info = {
                'category': 'Denied Topic',
                'purpose': 'Prevent political discussion',
                'filtering_type': 'Topic Filtering',
                'for_prompts_or_responses': 'Both',
                'strength_of_filter': 'Medium',
                'example': text[:100],
                'response_message': 'Political content detected'
            }
            self.db.store_guardrail_hit(user_id, tenant_id, guardrail_info)
            return False
        return True

    def check_religion(self, text, user_id, tenant_id):
        religion_keywords = ['religion', 'god', 'jesus', 'allah', 'buddha', 'hindu', 'christian', 'muslim', 'jewish']
        if any(keyword in text.lower() for keyword in religion_keywords):
            guardrail_info = {
                'category': 'Denied Topic',
                'purpose': 'Prevent religious discussion',
                'filtering_type': 'Topic Filtering',
                'for_prompts_or_responses': 'Both',
                'strength_of_filter': 'Medium',
                'example': text[:100],
                'response_message': 'Religious content detected'
            }
            self.db.store_guardrail_hit(user_id, tenant_id, guardrail_info)
            return False
        return True

    def check_legal_advice(self, text, user_id, tenant_id):
        legal_keywords = ['legal advice', 'lawsuit', 'sue', 'court', 'lawyer', 'attorney', 'legal action']
        if any(keyword in text.lower() for keyword in legal_keywords):
            guardrail_info = {
                'category': 'Denied Topic',
                'purpose': 'Prevent legal advice',
                'filtering_type': 'Topic Filtering',
                'for_prompts_or_responses': 'Both',
                'strength_of_filter': 'Medium',
                'example': text[:100],
                'response_message': 'Legal advice content detected'
            }
            self.db.store_guardrail_hit(user_id, tenant_id, guardrail_info)
            return False
        return True
    
    def check_profanity(self, text, user_id, tenant_id):
        bad_words = [
            "shit", "damn", "fuck", "ass", "bitch", "bastard", "cunt",
            "dick", "pussy", "asshole", "motherfucker", "crap",
            "piss", "hell", "whore", "slut", "twat", "wanker",
            "faggot", "dyke", "nigger", "spic", "chink", "kike",
            "retard", "cripple", "gimp", "idiot", "moron",
            "dumbass", "jackass", "bastard", "son of a bitch",
            "damn it", "goddamn", "holy shit",
            # Add more words to the list as needed
        ]
        for word in bad_words:
            if word in text.lower():
                guardrail_info = {
                    'category': 'Word Filter',
                    'purpose': 'Prevent profanity',
                    'filtering_type': 'Content Filtering',
                    'for_prompts_or_responses': 'Both',
                    'strength_of_filter': 'High',
                    'example': text[:100],
                    'response_message': f"Profanity detected: '{word}'" 
                }
                self.db.store_guardrail_hit(user_id, tenant_id, guardrail_info)
                return False 
        return True
    
    def check_pii(self, text, user_id, tenant_id):
        pii_patterns = [
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email
            r"\d{3}-\d{2}-\d{4}",  # SSN (US)
            r"\([0-9]{3}\) [0-9]{3}-\d{4}",  # Phone number (US)
            r"\d{16}",  # Credit card number
            r"\w{2}-\d{7}"  # Passport number (US)
        ]
        for pattern in pii_patterns:
            if re.search(pattern, text):
                guardrail_info = {
                    'category': 'Sensitive Information',
                    'purpose': 'Protect PII',
                    'filtering_type': 'Content Filtering',
                    'for_prompts_or_responses': 'Both',
                    'strength_of_filter': 'High',
                    'example': text[:100],
                    'response_message': "PII detected and masked"
                }
                self.db.store_guardrail_hit(user_id, tenant_id, guardrail_info)
                return False
        return True


def process_input(text: str, user_id: str, tenant_id: str) -> tuple:
    """Process user input through guardrails"""
    guardrails = Guardrails()
    
    try:
        # Content checks
        if not guardrails.check_hate_speech(text, user_id, tenant_id):
            return False, "I apologize, but I cannot process that type of content."
        if not guardrails.check_insult(text, user_id, tenant_id):
            return False, "I apologize, but I cannot process that type of content."
        if not guardrails.check_sexual_content(text, user_id, tenant_id):
            return False, "I apologize, but I cannot process that type of content."
        if not guardrails.check_violence(text, user_id, tenant_id):
            return False, "I apologize, but I cannot process that type of content."
        if not guardrails.check_misconduct(text, user_id, tenant_id):
            return False, "I apologize, but I cannot process that type of content."

        # Profanity check
        if not guardrails.check_profanity(text, user_id, tenant_id):
            return False, "I apologize, but I cannot process that type of content."

        # PII check
        if not guardrails.check_pii(text, user_id, tenant_id):
            return False, "I cannot process requests containing sensitive information."

        # Denied topics check
        if not guardrails.check_politics(text, user_id, tenant_id):
            return False, "I apologize, but I cannot discuss that topic."
        if not guardrails.check_religion(text, user_id, tenant_id):
            return False, "I apologize, but I cannot discuss that topic."
        if not guardrails.check_legal_advice(text, user_id, tenant_id):
            return False, "I apologize, but I cannot discuss that topic."

        return True, None

    except Exception as e:
        logging.error(f"Error in process_input: {str(e)}")
        return False, "An error occurred while processing your request."

def generate_response(text: str, user_id: str, tenant_id: str) -> tuple:
    """Generate response using Gemini and check through guardrails"""
    try:
        # First check if input is allowed
        is_safe, message = process_input(text, user_id, tenant_id)
        if not is_safe:
            return message, None

        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            text,
            generation_config={
                "max_output_tokens": 300,
                "temperature": 0.7
            }
        )
        
        generated_text = response.text

        # Check generated response through guardrails
        is_safe, message = process_input(generated_text, user_id, tenant_id)
        if not is_safe:
            return "I apologize, but I need to provide a different response.", None

        return generated_text, None

    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error processing your request.", None

def main():
    try:
        user_id = "test_user"  # In practice, get this from your auth system
        tenant_id = "test_tenant"  # In practice, get this from your auth system
        
        user_input = input("Enter your text: ")
        response, _ = generate_response(user_input, user_id, tenant_id)
        print("\nResponse:", response)

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print("An error occurred while processing your request.")

if __name__ == '__main__':
    main()