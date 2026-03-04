import sys
import os
sys.path.append('/home/cdac/Downloads/India-AI-Intelligent-Document-Processing-main/surya')

from PIL import Image
import json

class OCRProcessor:
    def __init__(self):
        pass
        
    def process_digital_pdf(self, pdf_path):
        """Process digital PDF using Surya OCR (treat as scanned)"""
        return self.process_scanned_document(pdf_path)
    
    def process_scanned_document(self, file_path):
        """Process any document using Surya OCR"""
        try:
            # Import surya modules
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
            from surya.foundation import FoundationPredictor
            from surya.common.surya.schema import TaskNames
            
            # Load models
            foundation_predictor = FoundationPredictor()
            det_predictor = DetectionPredictor()
            rec_predictor = RecognitionPredictor(foundation_predictor)
            
            # Load and preprocess image/PDF
            if file_path.lower().endswith('.pdf'):
                import pymupdf
                doc = pymupdf.open(file_path)
                images = []
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    preprocessed_img = img  # Use image directly
                    images.append(preprocessed_img)
                doc.close()
            else:
                original_image = Image.open(file_path)
                images = [original_image]
            
            # Run OCR
            task_names = [TaskNames.ocr_with_boxes] * len(images)
            predictions = rec_predictor(
                images,
                task_names=task_names,
                det_predictor=det_predictor,
                math_mode=False
            )
            
            # Format output with bboxes at root level
            result = {
                "text": "",
                "bboxes": [],  # ← THIS IS THE KEY FIX - bboxes at root level
                "metadata": {
                    "pages": len(images),
                    "layout_analysis": {
                        "sections": [],
                        "tables": [],
                        "headers": [],
                        "page_structure": []
                    }
                },
                "format": "json",
                "engine": "surya_ocr"
            }
            
            all_bboxes = []  # Collect all bboxes for root level
            
            for i, pred in enumerate(predictions):
                page_text = ""
                page_data = []
                bboxes_for_nav = []
                
                for text_line in pred.text_lines:
                    page_text += text_line.text + "\n"
                    
                    # Extract bbox coordinates
                    bbox = text_line.bbox
                    
                    bbox_data = {
                        "text": text_line.text,
                        "bbox": bbox,
                        "confidence": getattr(text_line, 'confidence', 0.9),
                        "page": i  # Add page number
                    }
                    page_data.append(bbox_data)
                    
                    # Add to root-level bboxes (this is what the app needs!)
                    all_bboxes.append({
                        "bbox": bbox,  # [x1, y1, x2, y2] format
                        "text": text_line.text,
                        "page": i,
                        "confidence": getattr(text_line, 'confidence', 0.9)
                    })
                    
                    # Add to navigation metadata
                    bboxes_for_nav.append({
                        "type": "text_line",
                        "page": i + 1,
                        "bbox": bbox,
                        "text_preview": text_line.text[:50]
                    })
                
                result["text"] += f"## Page {i+1}\n{page_text}\n"
                result["metadata"]["layout_analysis"]["page_structure"].append({
                    "page_number": i+1,
                    "text_lines": page_data,
                    "bboxes": bboxes_for_nav
                })
            
            # Assign all collected bboxes to root level
            result["bboxes"] = all_bboxes
            
            return result
            
        except Exception as e:
            print(f"Error processing document with Surya: {e}")
            import traceback
            traceback.print_exc()
            return None
