"""
Template Matching Module for Number Detection

Uses OpenCV template matching to find specific numbers in images.
Complements OCR by providing a visual matching approach that can work
even when text recognition fails.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageFont
import tempfile
import easyocr
from fuzzy_matcher import FuzzyMatcher


class TemplateGenerator:
    """Generate number templates for template matching"""
    
    def __init__(self, template_size: Tuple[int, int] = (50, 80)):
        """
        Initialize template generator
        
        Args:
            template_size: Size of generated templates (width, height)
        """
        self.template_size = template_size
    
    def generate_number_template(self, number: str, 
                                font_size: int = 60,
                                font_color: int = 0) -> np.ndarray:
        """
        Generate a template image for a specific number
        
        Args:
            number: Number string to generate template for
            font_size: Size of font to use
            font_color: Color of text (0=black, 255=white)
            
        Returns:
            OpenCV image array (grayscale) of the number template
        """
        # Create white background
        img = Image.new('L', self.template_size, color=255)
        draw = ImageDraw.Draw(img)
        
        # Try to use a standard font, fallback to default
        try:
            # Try to use Arial or similar font
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",          # Windows
                "/System/Library/Fonts/Arial.ttf",     # macOS
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
            ]
            
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), number, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text
        x = (self.template_size[0] - text_width) // 2
        y = (self.template_size[1] - text_height) // 2
        
        # Draw text
        draw.text((x, y), number, font=font, fill=font_color)
        
        # Convert to OpenCV format
        cv_img = np.array(img)
        
        return cv_img
    
    def generate_multiple_styles(self, number: str) -> List[np.ndarray]:
        """
        Generate multiple template styles for better matching
        
        Args:
            number: Number string to generate templates for
            
        Returns:
            List of template images in different styles
        """
        templates = []
        
        # Different font sizes
        for font_size in [40, 50, 60, 70]:
            template = self.generate_number_template(number, font_size, 0)
            templates.append(template)
        
        # Different styles (inverted colors) - white text on black background
        # Create black text on white background, then invert to get white on black
        template_inverted = self.generate_number_template(number, 60, 0)
        template_inverted = 255 - template_inverted  # Invert: white->black, black->white
        templates.append(template_inverted)
        
        return templates


class TemplateMatcher:
    """Template matching for number detection in images"""
    
    def __init__(self, template_size: Tuple[int, int] = (50, 80)):
        """
        Initialize template matcher
        
        Args:
            template_size: Size of templates to generate
        """
        self.template_generator = TemplateGenerator(template_size)
        self.template_cache = {}  # Cache generated templates
        self.ocr_reader = None  # Lazy load OCR
        self.fuzzy_matcher = FuzzyMatcher(enable_ocr_corrections=True)
    
    def get_templates(self, number: str) -> List[np.ndarray]:
        """
        Get or generate templates for a specific number
        
        Args:
            number: Number string to get templates for
            
        Returns:
            List of template images
        """
        if number not in self.template_cache:
            self.template_cache[number] = self.template_generator.generate_multiple_styles(number)
        
        return self.template_cache[number]
    
    def match_template_single(self, image: np.ndarray, template: np.ndarray,
                            method: int = cv2.TM_CCOEFF_NORMED) -> Dict:
        """
        Perform template matching with a single template
        
        Args:
            image: Input image to search in
            template: Template to search for
            method: OpenCV template matching method
            
        Returns:
            Dictionary with match results
        """
        # Ensure both images are grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        result = cv2.matchTemplate(image, template, method)
        
        # Get best match location and confidence
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # For SQDIFF methods, lower values are better matches
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            confidence = 1 - min_val
            match_location = min_loc
        else:
            confidence = max_val
            match_location = max_loc
        
        # Get template dimensions for bounding box
        h, w = template.shape[:2]
        
        return {
            'confidence': float(confidence),
            'location': match_location,
            'bounding_box': {
                'top_left': match_location,
                'bottom_right': (match_location[0] + w, match_location[1] + h),
                'width': w,
                'height': h
            },
            'result_matrix': result
        }
    
    def find_multiple_matches(self, image: np.ndarray, template: np.ndarray,
                            threshold: float = 0.8,
                            method: int = cv2.TM_CCOEFF_NORMED) -> List[Dict]:
        """
        Find multiple matches of a template in an image
        
        Args:
            image: Input image to search in
            template: Template to search for
            threshold: Minimum confidence threshold for matches
            method: OpenCV template matching method
            
        Returns:
            List of match dictionaries
        """
        # Ensure both images are grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        result = cv2.matchTemplate(image, template, method)
        
        # Find all locations above threshold
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            locations = np.where(result <= 1 - threshold)
            confidences = 1 - result[locations]
        else:
            locations = np.where(result >= threshold)
            confidences = result[locations]
        
        # Get template dimensions
        h, w = template.shape[:2]
        
        # Create list of matches
        matches = []
        for i, (y, x) in enumerate(zip(locations[0], locations[1])):
            match = {
                'confidence': float(confidences[i]),
                'location': (x, y),
                'bounding_box': {
                    'top_left': (x, y),
                    'bottom_right': (x + w, y + h),
                    'width': w,
                    'height': h
                }
            }
            matches.append(match)
        
        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return matches
    
    def check_for_number(self, image_path: str, expected_number: str,
                        threshold: float = 0.7) -> Dict:
        """
        Check if an expected number appears in an image using enhanced template matching
        
        Args:
            image_path: Path to image file
            expected_number: Expected number to search for
            threshold: Minimum confidence threshold for match
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Get templates for the expected number
            templates = self.get_templates(expected_number)
            
            # Enhanced ensemble scoring using multiple methods
            ensemble_result = self._ensemble_template_matching(image, templates, expected_number)
            
            # OCR validation if template confidence is reasonable
            if ensemble_result['ensemble_confidence'] >= 0.5:  # Lower threshold for OCR check
                ocr_validation = self._validate_with_ocr(image, ensemble_result, expected_number)
                
                # Combine template and OCR confidence intelligently
                if ocr_validation['ocr_found']:
                    # OCR confirms: boost confidence
                    final_confidence = min(1.0, ensemble_result['ensemble_confidence'] * 1.2)
                else:
                    # Determine why OCR failed
                    reason = ocr_validation.get('reason', '')
                    ocr_text = ocr_validation.get('ocr_text', '')
                    
                    if reason == 'no_matching_text' and len(ocr_text.strip()) > 0:
                        # OCR read something but it's not the expected number
                        # This is strong negative evidence - significantly reduce confidence
                        final_confidence = ensemble_result['ensemble_confidence'] * 0.2
                    elif reason in ['region_too_small', 'ocr_error', 'no_bounding_box']:
                        # OCR couldn't run properly - mild penalty only
                        final_confidence = ensemble_result['ensemble_confidence'] * 0.8
                    else:
                        # OCR found nothing or unclear result - mild penalty
                        final_confidence = ensemble_result['ensemble_confidence'] * 0.7
                    
                ensemble_result['ensemble_confidence'] = final_confidence
                ensemble_result['ocr_validation'] = ocr_validation
            else:
                ensemble_result['ocr_validation'] = {'ocr_found': False, 'reason': 'template_confidence_too_low'}
            
            # Apply threshold
            found = ensemble_result['ensemble_confidence'] >= threshold
            
            result = {
                'found': found,
                'confidence': ensemble_result['ensemble_confidence'],
                'expected_number': expected_number,
                'location': ensemble_result['location'] if found else None,
                'bounding_box': ensemble_result['bounding_box'] if found else None,
                'template_index': ensemble_result['best_template_idx'],
                'method': 'enhanced_template_matching',
                'threshold_used': threshold,
                'ensemble_scores': ensemble_result['method_scores'],
                'consensus_score': ensemble_result.get('consensus_score', 0.0)
            }
            
            return result
            
        except Exception as e:
            return {
                'found': False,
                'confidence': 0.0,
                'expected_number': expected_number,
                'location': None,
                'bounding_box': None,
                'template_index': -1,
                'method': 'enhanced_template_matching',
                'threshold_used': threshold,
                'error': str(e)
            }
    
    def _ensemble_template_matching(self, image: np.ndarray, templates: List[np.ndarray], 
                                   expected_number: str) -> Dict:
        """
        Enhanced template matching using ensemble scoring from multiple methods
        
        Args:
            image: Input image
            templates: List of templates to match
            expected_number: Expected number for validation
            
        Returns:
            Dictionary with ensemble matching results
        """
        # Different matching methods with weights
        methods = [
            (cv2.TM_CCOEFF_NORMED, 0.4),    # Good for contrast differences
            (cv2.TM_CCORR_NORMED, 0.3),     # Good for brightness matching  
            (cv2.TM_SQDIFF_NORMED, 0.3),    # Good for exact shape matching
        ]
        
        best_ensemble_score = 0.0
        best_result = None
        best_template_idx = -1
        method_scores = {}
        
        # Test each template with all methods
        for template_idx, template in enumerate(templates):
            template_scores = []
            template_results = []
            
            for method, weight in methods:
                match_result = self.match_template_single(image, template, method)
                confidence = match_result['confidence']
                template_scores.append(confidence * weight)
                template_results.append(match_result)
            
            # Calculate ensemble score for this template
            ensemble_score = sum(template_scores)
            
            # Penalize if scores are too inconsistent (likely false positive)
            score_std = np.std([r['confidence'] for r in template_results])
            if score_std > 0.3:  # High variance indicates unreliable match
                ensemble_score *= 0.7  # Reduce confidence
            
            method_scores[f'template_{template_idx}'] = {
                'individual_scores': [r['confidence'] for r in template_results],
                'ensemble_score': ensemble_score,
                'score_std': score_std
            }
            
            if ensemble_score > best_ensemble_score:
                best_ensemble_score = ensemble_score
                best_result = template_results[0]  # Use first method for location
                best_template_idx = template_idx
        
        # Calculate consensus score (how well templates agree)
        template_ensemble_scores = [method_scores[f'template_{i}']['ensemble_score'] 
                                  for i in range(len(templates))]
        consensus_score = 1.0 - (np.std(template_ensemble_scores) / (np.mean(template_ensemble_scores) + 1e-6))
        consensus_score = max(0.0, min(1.0, consensus_score))  # Clamp to [0,1]
        
        # Adjust final confidence based on consensus
        final_confidence = best_ensemble_score * (0.7 + 0.3 * consensus_score)
        
        return {
            'ensemble_confidence': final_confidence,
            'location': best_result['location'] if best_result else None,
            'bounding_box': best_result['bounding_box'] if best_result else None,
            'best_template_idx': best_template_idx,
            'method_scores': method_scores,
            'consensus_score': consensus_score
        }
    
    def _validate_with_ocr(self, image: np.ndarray, ensemble_result: Dict, expected_number: str) -> Dict:
        """
        Validate template match using OCR with robust region extraction and preprocessing
        
        Args:
            image: Original image
            ensemble_result: Result from ensemble template matching
            expected_number: Expected number to validate
            
        Returns:
            Dictionary with OCR validation results
        """
        try:
            # Initialize OCR reader if needed
            if self.ocr_reader is None:
                self.ocr_reader = easyocr.Reader(['en'], verbose=False)
            
            # Strategy 1: Use full image OCR to validate the general area
            full_ocr_results = self.ocr_reader.readtext(image, detail=1)  # detail=1 for coordinates
            
            # Check if expected number exists anywhere in the full image
            for bbox_coords, ocr_text, confidence in full_ocr_results:
                clean_text = ''.join(filter(str.isdigit, str(ocr_text)))
                
                if clean_text == expected_number and confidence > 0.8:
                    # Found exact match in full image with high confidence
                    template_bbox = ensemble_result['bounding_box']
                    if template_bbox:
                        # Check if OCR bbox overlaps with template bbox
                        ocr_center_x = (bbox_coords[0][0] + bbox_coords[2][0]) // 2
                        ocr_center_y = (bbox_coords[0][1] + bbox_coords[2][1]) // 2
                        template_center_x = (template_bbox['top_left'][0] + template_bbox['bottom_right'][0]) // 2
                        template_center_y = (template_bbox['top_left'][1] + template_bbox['bottom_right'][1]) // 2
                        
                        # Calculate distance between centers
                        distance = ((ocr_center_x - template_center_x) ** 2 + 
                                   (ocr_center_y - template_center_y) ** 2) ** 0.5
                        
                        # If centers are reasonably close (within 100 pixels), consider it a match
                        if distance < 100:
                            return {
                                'ocr_found': True,
                                'ocr_text': clean_text,
                                'fuzzy_score': 1.0,
                                'match_type': 'full_image_exact',
                                'ocr_confidence': float(confidence),
                                'distance_from_template': float(distance)
                            }
            
            # Strategy 2: Extract and process template region with multiple preprocessing
            bbox = ensemble_result['bounding_box']
            if not bbox:
                return {'ocr_found': False, 'reason': 'no_bounding_box', 'ocr_text': ''}
            
            # Try multiple region sizes and preprocessing approaches
            preprocessing_methods = [
                (20, 1.0, 0),      # Standard padding, no enhancement
                (30, 1.5, 10),     # Larger padding, slight contrast boost
                (40, 2.0, 20),     # Even larger padding, more contrast
            ]
            
            best_match = None
            best_score = 0.0
            
            for padding, scale_factor, contrast_boost in preprocessing_methods:
                # Extract region with current parameters
                x1 = max(0, bbox['top_left'][0] - padding)
                y1 = max(0, bbox['top_left'][1] - padding) 
                x2 = min(image.shape[1], bbox['bottom_right'][0] + padding)
                y2 = min(image.shape[0], bbox['bottom_right'][1] + padding)
                
                roi = image[y1:y2, x1:x2]
                
                # Skip if region is too small
                if roi.shape[0] < 10 or roi.shape[1] < 10:
                    continue
                
                # Apply preprocessing
                processed_roi = roi.copy()
                
                if contrast_boost > 0:
                    # Enhance contrast
                    processed_roi = cv2.convertScaleAbs(processed_roi, alpha=1.0 + contrast_boost/100.0, beta=0)
                
                if scale_factor > 1.0:
                    # Scale up for better OCR
                    new_width = int(processed_roi.shape[1] * scale_factor)
                    new_height = int(processed_roi.shape[0] * scale_factor)
                    processed_roi = cv2.resize(processed_roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                # Run OCR on processed region
                try:
                    ocr_results = self.ocr_reader.readtext(processed_roi, detail=0)
                    
                    for ocr_text in ocr_results:
                        clean_text = ''.join(filter(str.isdigit, str(ocr_text)))
                        
                        if clean_text == expected_number:
                            return {
                                'ocr_found': True,
                                'ocr_text': clean_text,
                                'fuzzy_score': 1.0,
                                'match_type': 'region_exact',
                                'preprocessing': f'padding={padding}, scale={scale_factor}, contrast={contrast_boost}'
                            }
                        
                        if len(clean_text) > 0:
                            fuzzy_score = self.fuzzy_matcher.calculate_similarity(clean_text, expected_number)
                            if fuzzy_score > best_score:
                                best_score = fuzzy_score
                                best_match = {
                                    'ocr_found': fuzzy_score >= 0.7,  # Lower threshold for region OCR
                                    'ocr_text': clean_text,
                                    'fuzzy_score': fuzzy_score,
                                    'match_type': 'region_fuzzy',
                                    'preprocessing': f'padding={padding}, scale={scale_factor}, contrast={contrast_boost}'
                                }
                except:
                    continue  # Skip this preprocessing method if it fails
            
            # Return best match if found
            if best_match and best_match['ocr_found']:
                return best_match
            
            # Strategy 3: Check if we found the expected number anywhere in full image
            # even if not close to template (for final validation)
            for bbox_coords, ocr_text, confidence in full_ocr_results:
                clean_text = ''.join(filter(str.isdigit, str(ocr_text)))
                if clean_text == expected_number and confidence > 0.5:
                    return {
                        'ocr_found': True,
                        'ocr_text': clean_text,
                        'fuzzy_score': 0.8,  # Moderate confidence since not near template
                        'match_type': 'full_image_distant',
                        'ocr_confidence': float(confidence)
                    }
            
            # No match found anywhere
            all_ocr_text = ' '.join([str(t) for _, t, _ in full_ocr_results])
            return {
                'ocr_found': False,
                'reason': 'no_matching_text',
                'ocr_text': all_ocr_text,
                'expected': expected_number,
                'best_fuzzy_score': best_score
            }
            
        except Exception as e:
            return {
                'ocr_found': False,
                'reason': 'ocr_error',
                'error': str(e),
                'ocr_text': ''
            }
    
    def visualize_match(self, image_path: str, match_result: Dict,
                       output_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Visualize template matching results by drawing bounding boxes
        
        Args:
            image_path: Path to original image
            match_result: Result from check_for_number()
            output_path: Optional path to save visualization
            
        Returns:
            Image with bounding boxes drawn (if match found)
        """
        try:
            # Load original image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            if match_result['found'] and match_result['bounding_box']:
                bbox = match_result['bounding_box']
                
                # Draw bounding box
                cv2.rectangle(image, 
                            bbox['top_left'],
                            bbox['bottom_right'],
                            (0, 255, 0),  # Green color
                            2)
                
                # Add confidence text
                conf_text = f"{match_result['confidence']:.3f}"
                text_pos = (bbox['top_left'][0], bbox['top_left'][1] - 10)
                cv2.putText(image, conf_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 0), 2)
            
            # Save if output path provided
            if output_path:
                cv2.imwrite(str(output_path), image)
            
            return image
            
        except Exception as e:
            print(f"Error visualizing match: {e}")
            return None


def demo():
    """Demonstration of template matching functionality"""
    matcher = TemplateMatcher()
    
    print("Template Matching Demo")
    print("=" * 30)
    
    # Generate templates for number "20"
    templates = matcher.get_templates("20")
    print(f"Generated {len(templates)} templates for number '20'")
    
    # Save templates for inspection
    template_dir = "templates_demo"
    os.makedirs(template_dir, exist_ok=True)
    
    for i, template in enumerate(templates):
        cv2.imwrite(f"{template_dir}/template_20_{i}.png", template)
        print(f"  Saved template_{i}: {template.shape}")
    
    print(f"\nTemplates saved to '{template_dir}/' directory")
    print("You can view these template images to see what the matcher looks for.")


if __name__ == "__main__":
    demo()