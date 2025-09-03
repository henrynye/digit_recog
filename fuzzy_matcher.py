"""
Fuzzy Matching Module for OCR Text Similarity

Provides string similarity scoring using various algorithms to handle common OCR errors
and variations in detected text compared to expected values.
"""

from fuzzywuzzy import fuzz
from fuzzywuzzy.process import extractOne
import re
from typing import Dict, List, Tuple, Optional


class FuzzyMatcher:
    """Fuzzy string matching with OCR-specific optimizations"""
    
    # Common OCR misreads - maps incorrect OCR characters to correct ones
    OCR_CORRECTIONS = {
        'O': '0',    # Letter O confused with digit 0
        'o': '0',    # Lowercase o confused with digit 0
        'l': '1',    # Lowercase L confused with digit 1
        'I': '1',    # Uppercase I confused with digit 1
        '|': '1',    # Pipe character confused with digit 1
        'S': '5',    # Letter S confused with digit 5
        's': '5',    # Lowercase s confused with digit 5
        'Z': '2',    # Letter Z confused with digit 2
        'g': '9',    # Lowercase g confused with digit 9
        'G': '6',    # Uppercase G confused with digit 6
        'b': '6',    # Lowercase b confused with digit 6
        'B': '8',    # Uppercase B confused with digit 8
        'D': '0',    # Uppercase D confused with digit 0
    }
    
    def __init__(self, enable_ocr_corrections: bool = True):
        """
        Initialize fuzzy matcher
        
        Args:
            enable_ocr_corrections: Whether to apply OCR error corrections
        """
        self.enable_ocr_corrections = enable_ocr_corrections
    
    def correct_ocr_errors(self, text: str) -> str:
        """
        Apply common OCR error corrections to text
        
        Args:
            text: Input text that may contain OCR errors
            
        Returns:
            Corrected text with common OCR errors fixed
        """
        if not self.enable_ocr_corrections:
            return text
            
        corrected = text
        for wrong, correct in self.OCR_CORRECTIONS.items():
            corrected = corrected.replace(wrong, correct)
        
        return corrected
    
    def normalize_number_string(self, text: str) -> str:
        """
        Normalize a string to extract and clean number portions
        
        Args:
            text: Input text containing numbers
            
        Returns:
            Normalized string with only digits, spaces, and common separators
        """
        # Apply OCR corrections first
        corrected = self.correct_ocr_errors(text)
        
        # Extract digits and common separators
        normalized = re.sub(r'[^0-9\s\-\.]', '', corrected)
        
        # Remove extra spaces and normalize
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def calculate_similarity(self, text1: str, text2: str, 
                           method: str = 'ratio') -> float:
        """
        Calculate similarity between two strings using specified method
        
        Args:
            text1: First string to compare
            text2: Second string to compare
            method: Similarity method ('ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio')
            
        Returns:
            Similarity score between 0.0 (no match) and 1.0 (perfect match)
        """
        # Normalize both strings
        norm_text1 = self.normalize_number_string(str(text1))
        norm_text2 = self.normalize_number_string(str(text2))
        
        # Choose similarity calculation method
        if method == 'ratio':
            score = fuzz.ratio(norm_text1, norm_text2)
        elif method == 'partial_ratio':
            score = fuzz.partial_ratio(norm_text1, norm_text2)
        elif method == 'token_sort_ratio':
            score = fuzz.token_sort_ratio(norm_text1, norm_text2)
        elif method == 'token_set_ratio':
            score = fuzz.token_set_ratio(norm_text1, norm_text2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        # Convert from 0-100 to 0.0-1.0
        return score / 100.0
    
    def find_best_match(self, target: str, candidates: List[str], 
                       threshold: float = 0.6) -> Optional[Dict]:
        """
        Find the best match for target string among candidates
        
        Args:
            target: Target string to match against
            candidates: List of candidate strings
            threshold: Minimum similarity threshold
            
        Returns:
            Dictionary with match results or None if no match above threshold
            {
                'match': str,           # Best matching candidate
                'similarity': float,    # Similarity score (0.0-1.0)
                'normalized_target': str,     # Normalized target string
                'normalized_match': str       # Normalized match string
            }
        """
        if not candidates:
            return None
        
        # Normalize target
        normalized_target = self.normalize_number_string(target)
        
        # Normalize all candidates and calculate similarities
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            similarity = self.calculate_similarity(target, candidate)
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = {
                    'match': candidate,
                    'similarity': similarity,
                    'normalized_target': normalized_target,
                    'normalized_match': self.normalize_number_string(candidate)
                }
        
        return best_match
    
    def check_expected_number(self, expected: str, detected_texts: List[str],
                             threshold: float = 0.8) -> Dict:
        """
        Check if an expected number appears in detected texts
        
        Args:
            expected: Expected number/text to find
            detected_texts: List of detected text strings
            threshold: Minimum similarity threshold for a match
            
        Returns:
            Dictionary with detection results:
            {
                'found': bool,              # Whether expected number was found
                'best_match': str or None,  # Best matching detected text
                'similarity': float,        # Similarity score of best match
                'all_matches': List[Dict],  # All matches above threshold
                'method': str              # Detection method used
            }
        """
        if not detected_texts:
            return {
                'found': False,
                'best_match': None,
                'similarity': 0.0,
                'all_matches': [],
                'method': 'fuzzy_matching'
            }
        
        # Find exact matches first (after normalization)
        normalized_expected = self.normalize_number_string(expected)
        exact_matches = []
        
        for text in detected_texts:
            normalized_text = self.normalize_number_string(text)
            if normalized_expected == normalized_text:
                exact_matches.append({
                    'text': text,
                    'similarity': 1.0,
                    'normalized': normalized_text
                })
        
        if exact_matches:
            best_exact = max(exact_matches, key=lambda x: len(x['text']))
            return {
                'found': True,
                'best_match': best_exact['text'],
                'similarity': 1.0,
                'all_matches': exact_matches,
                'method': 'exact_match'
            }
        
        # No exact matches, use fuzzy matching
        all_matches = []
        best_score = 0.0
        best_match_text = None
        
        for text in detected_texts:
            similarity = self.calculate_similarity(expected, text)
            
            if similarity >= threshold:
                match_info = {
                    'text': text,
                    'similarity': similarity,
                    'normalized': self.normalize_number_string(text)
                }
                all_matches.append(match_info)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match_text = text
        
        return {
            'found': len(all_matches) > 0,
            'best_match': best_match_text,
            'similarity': best_score,
            'all_matches': all_matches,
            'method': 'fuzzy_matching'
        }
    
    def get_similarity_breakdown(self, text1: str, text2: str) -> Dict:
        """
        Get detailed similarity breakdown using multiple methods
        
        Args:
            text1: First string to compare
            text2: Second string to compare
            
        Returns:
            Dictionary with similarity scores from different methods
        """
        methods = ['ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio']
        
        breakdown = {
            'original_text1': text1,
            'original_text2': text2,
            'normalized_text1': self.normalize_number_string(text1),
            'normalized_text2': self.normalize_number_string(text2),
            'scores': {}
        }
        
        for method in methods:
            breakdown['scores'][method] = self.calculate_similarity(text1, text2, method)
        
        # Overall score (average of ratio and partial_ratio)
        breakdown['overall_score'] = (
            breakdown['scores']['ratio'] + breakdown['scores']['partial_ratio']
        ) / 2.0
        
        return breakdown


def demo():
    """Demonstration of fuzzy matching functionality"""
    matcher = FuzzyMatcher()
    
    # Test OCR corrections
    print("OCR Error Corrections:")
    test_texts = ["2O", "l5", "3O8", "I23", "5G"]
    for text in test_texts:
        corrected = matcher.correct_ocr_errors(text)
        print(f"  '{text}' -> '{corrected}'")
    
    # Test similarity calculations
    print("\nSimilarity Tests:")
    expected = "234"
    detected = ["234", "Z34", "23O", "l234", "wrong"]
    
    for text in detected:
        result = matcher.check_expected_number(expected, [text], threshold=0.5)
        print(f"  Expected: '{expected}', Detected: '{text}' -> "
              f"Found: {result['found']}, Similarity: {result['similarity']:.3f}")


if __name__ == "__main__":
    demo()