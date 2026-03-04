import streamlit as st
import tempfile
import os
import json
from PIL import Image, ImageDraw, ImageFont
import pymupdf
import io
import pandas as pd
from datetime import datetime
import sys
import re
import base64

# Add paths
sys.path.insert(0, '/home/cdac/Downloads/India-AI-Intelligent-Document-Processing-main/marker')
sys.path.insert(0, '/home/cdac/Downloads/India-AI-Intelligent-Document-Processing-main/surya')

from database import DatabaseManager
from deep_translator import GoogleTranslator

# Safe Groq import
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# Professional Page Configuration
st.set_page_config(
    page_title="AI Document Intelligent Platform",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .header-subtitle {
        color: #e0e0e0;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #2a5298;
        color: #333;
    }
    
    .info-card strong {
        color: #1e3c72;
        font-weight: 600;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 3px 6px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .status-success {
        background-color: #10b981;
        color: white;
    }
    
    .status-processing {
        background-color: #f59e0b;
        color: white;
    }
    
    .status-info {
        background-color: #3b82f6;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border-radius: 6px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(42, 82, 152, 0.4);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 1.5rem;
        font-weight: 600;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

class DocumentViewer:
    """Simplified document viewer for professional presentation"""
    
    def __init__(self):
        self.font_size = 12
    
    def draw_red_bboxes(self, image, bboxes, current_page=0):
        """Draw red bounding boxes from OCR on the image"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Filter bboxes for current page
        page_bboxes = []
        for bbox_item in bboxes:
            if isinstance(bbox_item, dict):
                # Check if this bbox is for the current page
                bbox_page = bbox_item.get('page', 0)
                if bbox_page == current_page:
                    bbox_coords = bbox_item.get('bbox', [])
                    if len(bbox_coords) >= 4:
                        page_bboxes.append(bbox_coords)
            elif isinstance(bbox_item, (list, tuple)) and len(bbox_item) >= 4:
                page_bboxes.append(bbox_item)
        
        # Draw red boxes
        for bbox in page_bboxes:
            x1, y1, x2, y2 = bbox[:4]
            # Draw red bounding box with thicker line for visibility
            draw.rectangle([x1, y1, x2, y2], outline="#FF0000", width=3)
        
        return img_copy
    
    def display_document_simple(self, image, bboxes=None, current_page=0):
        """Display document with red OCR bounding boxes"""
        if bboxes and len(bboxes) > 0:
            image_with_boxes = self.draw_red_bboxes(image, bboxes, current_page)
            st.image(image_with_boxes, use_container_width=True)
        else:
            st.image(image, use_container_width=True)

class AdvancedOCRProcessor:
    def __init__(self, groq_api_key):
        self.db = DatabaseManager()
        self.groq_client = None
        self.viewer = DocumentViewer()
        
        # Initialize OCR processor ONCE here (loads models once)
        print("🚀 Initializing OCR models...")
        from ocr_processor import OCRProcessor
        self.ocr_processor = OCRProcessor()
        print("✅ OCR ready!")
        
        if GROQ_AVAILABLE and groq_api_key != "dummy":
            try:
                self.groq_client = Groq(api_key=groq_api_key)
            except:
                self.groq_client = None
        
    def store_input_files(self, uploaded_files):
        """Store uploaded files in input_table"""
        file_ids = []
        for file in uploaded_files:
            file.seek(0)
            file_data = {
                'filename': file.name,
                'file_data': file.read(),
                'file_type': file.name.split('.')[-1].lower(),
                'file_size': len(file.getvalue()),
                'upload_time': datetime.now(),
                'status': 'uploaded'
            }
            file.seek(0)
            file_id = self.db.db['input_table'].insert_one(file_data).inserted_id
            file_ids.append(str(file_id))
        return file_ids
    
    def process_with_surya_ocr(self, file_path, filename):
        """Process document with Surya OCR - multilingual text detection and recognition"""
        try:
            # Use the already-initialized OCR processor (models already loaded!)
            if file_path.lower().endswith('.pdf'):
                result = self.ocr_processor.process_digital_pdf(file_path)
            else:
                result = self.ocr_processor.process_scanned_document(file_path)
            
            if result and 'text' in result:
                if 'bboxes' not in result:
                    result['bboxes'] = []
                
                return {
                    'success': True,
                    'raw_text': result['text'],
                    'extracted_text': result['text'],
                    'metadata': result.get('metadata', {}),
                    'format': result.get('format', 'unknown'),
                    'bboxes': result.get('bboxes', []),
                    'engine': result.get('engine', 'surya_ocr')
                }
            else:
                return {'success': False, 'error': 'No text extracted'}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze_with_groq(self, raw_text):
        """Analyze text with Groq LLM and extract entities"""
        if not self.groq_client:
            return {
                "entities": {},
                "summary": "Analysis not available",
                "language": "unknown",
                "document_type": "unknown"
            }

        prompt = f"""
Analyze the following text and extract structured information.
Return ONLY valid JSON in this exact format:

{{
    "entities": {{
        "names": [],
        "aadhaar_numbers": [],
        "pan_numbers": [],
        "phone_numbers": [],
        "addresses": [],
        "dates": [],
        "amounts": [],
        "organizations": []
    }},
    "summary": "",
    "language": "",
    "document_type": ""
}}

Text to analyze:
{raw_text[:3000]}
"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )

            content = response.choices[0].message.content
            
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)

        except json.JSONDecodeError as e:
            return {
                "entities": {},
                "summary": f"JSON parsing failed: {str(e)}",
                "language": "unknown",
                "document_type": "unknown"
            }
        except Exception as e:
            return {
                "entities": {},
                "summary": f"Analysis failed: {str(e)}",
                "language": "unknown",
                "document_type": "unknown"
            }
    
    def calculate_entity_stats(self, entities):
        """Calculate total entities (with duplicates) and unique entities"""
        total_entities = 0  # Count ALL entities including duplicates
        unique_entities = set()  # Count only unique entities
        
        for field, values in entities.items():
            if values and isinstance(values, list):
                for value in values:
                    if value:  # Skip empty values
                        total_entities += 1  # Count each occurrence
                        # Normalize for uniqueness check (case-insensitive)
                        unique_entities.add(str(value).strip().lower())
        
        return {
            'total_count': total_entities,      # ALL entities (duplicates counted)
            'unique_count': len(unique_entities)  # Only UNIQUE entities
        }
    
    def is_english_text(self, text):
        """Check if text is primarily English"""
        if not text or len(text.strip()) == 0:
            return True
        
        english_chars = sum(1 for c in text if ord(c) < 128)
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        if total_chars == 0:
            return True
        
        return (english_chars / total_chars) > 0.80
    
    def detect_language(self, text):
        """Detect language of text"""
        try:
            from langdetect import detect
            return detect(text)
        except:
            if any('\u0900' <= c <= '\u097F' for c in text):
                return 'hi'
            elif any('\u0B80' <= c <= '\u0BFF' for c in text):
                return 'ta'
            elif any('\u0C00' <= c <= '\u0C7F' for c in text):
                return 'te'
            else:
                return 'en'
    
    def translate_line_by_line(self, text):
        """Translate text line by line"""
        lines = text.split('\n')
        translated_lines = []
        translation_stats = {
            'total_lines': len(lines),
            'translated_lines': 0,
            'english_lines': 0,
            'languages_detected': set()
        }
        
        for line in lines:
            line = line.strip()
            
            if not line:
                translated_lines.append('')
                continue
            
            if self.is_english_text(line):
                translated_lines.append(line)
                translation_stats['english_lines'] += 1
                translation_stats['languages_detected'].add('en')
                continue
            
            try:
                detected_lang = self.detect_language(line)
                translation_stats['languages_detected'].add(detected_lang)
                
                if detected_lang != 'en':
                    translator = GoogleTranslator(source=detected_lang, target='en')
                    translated = translator.translate(line)
                    translated_lines.append(translated)
                    translation_stats['translated_lines'] += 1
                else:
                    translated_lines.append(line)
                    translation_stats['english_lines'] += 1
                    
            except Exception as e:
                translated_lines.append(line)
        
        return {
            'translated_text': '\n'.join(translated_lines),
            'stats': {
                'total_lines': translation_stats['total_lines'],
                'translated_lines': translation_stats['translated_lines'],
                'english_lines': translation_stats['english_lines'],
                'languages_detected': list(translation_stats['languages_detected'])
            }
        }
    
    def translate_to_english(self, text):
        """Enhanced translation with line-by-line detection"""
        try:
            if not text or len(text.strip()) == 0:
                return {
                    "original_text": text,
                    "translated_text": text,
                    "detected_language": "unknown",
                    "confidence": 0.0,
                    "translation_stats": {}
                }
            
            translation_result = self.translate_line_by_line(text)
            
            return {
                "original_text": text,
                "translated_text": translation_result['translated_text'],
                "detected_language": "mixed",
                "confidence": 0.92,
                "translation_stats": translation_result['stats']
            }

        except Exception as e:
            return {
                "original_text": text,
                "translated_text": text,
                "detected_language": "unknown",
                "confidence": 0.0,
                "error": str(e),
                "translation_stats": {}
            }


def convert_pdf_to_images(pdf_path):
    """Convert PDF to images WITHOUT SCALING to match OCR coordinates"""
    images = []
    try:
        doc = pymupdf.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # CRITICAL FIX: Don't use Matrix(2,2) scaling - use 1:1 to match OCR
            pix = page.get_pixmap()  # NO SCALING
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        doc.close()
    except Exception as e:
        st.error(f"Error converting PDF: {str(e)}")
    return images


def main():
    # Professional Header with CDAC Branding
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📄 AI Document Intelligent Platform</h1>
        <p class="header-subtitle">Advanced OCR • Multi-Language Support • AI-Powered Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize processor
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") or "dummy"
    processor = AdvancedOCRProcessor(groq_api_key)
    
    # Simplified 3-tab interface
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Analysis Dashboard", "🔍 Document Search"])
    
    with tab1:
        st.markdown("### 📥 Upload Documents")
        st.markdown("*Supported formats: PDF, JPG, PNG, TIFF*")
        
        uploaded_files = st.file_uploader(
            "Choose files to process",
            type=['pdf', 'jpg', 'jpeg', 'png', 'tiff'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Files Uploaded</div>
                    <div class="metric-value">{len(uploaded_files)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_size = sum(len(f.getvalue()) for f in uploaded_files) / (1024 * 1024)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Size</div>
                    <div class="metric-value">{total_size:.2f} MB</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Status</div>
                    <div class="metric-value">✓ Ready</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Auto-store files
            if 'stored_file_ids' not in st.session_state or st.session_state.get('last_uploaded_count') != len(uploaded_files):
                with st.spinner("📥 Storing files..."):
                    file_ids = processor.store_input_files(uploaded_files)
                    st.session_state['stored_file_ids'] = file_ids
                    st.session_state['last_uploaded_count'] = len(uploaded_files)
                    st.success(f"✅ {len(file_ids)} files stored successfully")
            
            # Process button
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                temp_files = []
                
                for i, file in enumerate(uploaded_files):
                    status_text.info(f"⚙️ Processing: {file.name}")
                    
                    file.seek(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                        temp_files.append(tmp_path)
                    
                    try:
                        # OCR Processing with Surya
                        ocr_result = processor.process_with_surya_ocr(tmp_path, file.name)
                        
                        if ocr_result.get('success'):
                            raw_text = ocr_result['raw_text']
                            
                            # Translation
                            translation_result = processor.translate_to_english(raw_text)
                            
                            # Groq Analysis
                            groq_analysis = processor.analyze_with_groq(translation_result['translated_text'])
                            
                            # Calculate entity statistics
                            entity_stats = processor.calculate_entity_stats(groq_analysis.get('entities', {}))
                            
                            results.append({
                                'filename': file.name,
                                'file_path': tmp_path,
                                'ocr_result': ocr_result,
                                'groq_analysis': groq_analysis,
                                'translation': translation_result,
                                'entity_stats': entity_stats,
                                'success': True
                            })
                            
                            # Save to database with entity stats
                            processor.db.store_processed_document(
                                filename=file.name,
                                raw_text=raw_text,
                                translated_text=translation_result['translated_text'],
                                entities=groq_analysis.get('entities', {}),
                                summary=groq_analysis.get('summary', ''),
                                document_type=groq_analysis.get('document_type', ''),
                                detected_language=translation_result.get('detected_language', ''),
                                translation_stats=translation_result.get('translation_stats', {}),
                                entity_stats=entity_stats
                            )
                        else:
                            results.append({
                                'filename': file.name,
                                'file_path': tmp_path,
                                'error': ocr_result.get('error'),
                                'success': False
                            })
                    
                    except Exception as e:
                        results.append({
                            'filename': file.name,
                            'file_path': tmp_path,
                            'error': str(e),
                            'success': False
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.session_state['processing_results'] = results
                st.session_state['temp_files'] = temp_files
                status_text.success("✅ Processing complete!")
    
    with tab2:
        st.markdown("### 📊 Analysis Dashboard")
        
        if 'processing_results' in st.session_state:
            results = st.session_state['processing_results']
            
            # Document selector
            filenames = [r['filename'] for r in results]
            selected_file = st.selectbox("📄 Select document to view:", filenames)
            
            if selected_file:
                result_data = next(r for r in results if r['filename'] == selected_file)
                
                if result_data.get('success'):
                    groq_analysis = result_data['groq_analysis']
                    translation = result_data['translation']
                    entity_stats = result_data.get('entity_stats', {'total_count': 0, 'unique_count': 0})
                    
                    # Key Metrics Row - UPDATED WITH ENTITY METRICS
                    st.markdown("#### 📈 Document Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    trans_stats = translation.get('translation_stats', {})
                    with col1:
                        st.metric("Total Lines", trans_stats.get('total_lines', 0))
                    with col2:
                        # CHANGED: Total Entities (includes duplicates)
                        st.metric("Total Entities", entity_stats['total_count'])
                    with col3:
                        # CHANGED: Unique Entities
                        st.metric("Unique Entities Values", entity_stats['unique_count'])
                    with col4:
                        # Entity Categories (non-empty fields)
                        entities = groq_analysis.get('entities', {})
                        entity_categories = sum(1 for v in entities.values() if v)
                        st.metric("Entity Categories", entity_categories)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Main content in 2 columns
                    col_left, col_right = st.columns([1.2, 1])
                    
                    with col_left:
                        st.markdown("#### 📋 Extracted Entities")
                        
                        entity_data = []
                        for field, values in entities.items():
                            if values:
                                for value in values:
                                    entity_data.append({
                                        'Category': field.replace('_', ' ').title(),
                                        'Value': str(value)
                                    })
                        
                        if entity_data:
                            df = pd.DataFrame(entity_data)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No entities extracted from this document")
                        
                        # Document viewer with bounding boxes
                        st.markdown("#### 📄 Document Preview")
                        file_path = result_data['file_path']
                        
                        if os.path.exists(file_path):
                            if file_path.lower().endswith('.pdf'):
                                images = convert_pdf_to_images(file_path)
                            else:
                                images = [Image.open(file_path)]
                            
                            if images:
                                page_select = st.selectbox("Select page:", range(1, len(images) + 1))
                                bboxes = result_data['ocr_result'].get('bboxes', [])
                                # Pass current page index (0-based)
                                processor.viewer.display_document_simple(
                                    images[page_select - 1], 
                                    bboxes, 
                                    current_page=page_select - 1
                                )
                    
                    with col_right:
                        st.markdown("#### 📝 Document Information")
                        st.markdown(f"""
                        <div class="info-card">
                            <strong>Document Type:</strong> {groq_analysis.get('document_type', 'Unknown')}<br>
                            <strong>Language:</strong> {translation.get('detected_language', 'Unknown')}<br>
                            <strong>Processing Engine:</strong> {result_data['ocr_result'].get('engine', 'Unknown')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### 📄 Summary")
                        summary = groq_analysis.get('summary', 'No summary available')
                        st.markdown(f"""
                        <div class="info-card">
                            {summary}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### 🌐 Translated Text")
                        st.info("💡 **Note:** Raw OCR text appears unformatted as it's optimized for entity extraction. See the **Extracted Entities** table above for structured data.", icon="ℹ️")
                        translated_text = translation.get('translated_text', '')
                        # Show character count
                        char_count = len(translated_text)
                        st.caption(f"📝 {char_count:,} characters | Full text displayed below")
                        st.text_area(
                            "English Translation",
                            translated_text,  # FULL TEXT - no truncation
                            height=400,  # Increased height for scrolling
                            disabled=True,
                            label_visibility="collapsed"
                        )
                
                else:
                    st.error(f"❌ Processing failed: {result_data.get('error')}")
        else:
            st.info("👆 Upload and process documents to view analysis results")
    
    with tab3:
        st.markdown("### 🔍 Search Processed Documents")
        
        search_query = st.text_input(
            "🔎 Search in documents",
            placeholder="Enter keywords (e.g., Aadhaar, name, address...)"
        )
        
        if search_query:
            with st.spinner("Searching..."):
                query_regex = re.compile(re.escape(search_query.strip()), re.IGNORECASE)
                cursor = processor.db.db['processed_documents'].find({
                    "searchable_text": {"$regex": query_regex}
                })
                documents = list(cursor)
                
                if documents:
                    st.success(f"✅ Found {len(documents)} matching document(s)")
                    
                    for doc in documents:
                        with st.expander(f"📄 {doc['original_filename']} • {doc['processed_at'].strftime('%Y-%m-%d %H:%M')}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📝 Extracted Text**")
                                st.text_area(
                                    "Content",
                                    doc['translated_text'][:800],
                                    height=200,
                                    disabled=True,
                                    label_visibility="collapsed",
                                    key=f"text_{doc['_id']}"
                                )
                            
                            with col2:
                                st.markdown("**📋 Key Information**")
                                
                                # Get entity stats from document
                                entity_stats = doc.get('entity_stats', {'total_count': 0, 'unique_count': 0})
                                
                                st.markdown(f"""
                                <div class="info-card">
                                    <strong>Type:</strong> {doc.get('document_type', 'Unknown')}<br>
                                    <strong>Language:</strong> {doc.get('detected_language', 'Unknown')}<br>
                                    <strong>Total Entities:</strong> {entity_stats.get('total_count', 0)}<br>
                                    <strong>Unique Entities Values:</strong> {entity_stats.get('unique_count', 0)}<br>
                                    <strong>Summary:</strong> {doc.get('summary', 'N/A')[:100]}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show entities
                                entities = doc.get('entities', {})
                                entity_list = []
                                for field, values in entities.items():
                                    if values:
                                        for val in values:
                                            entity_list.append({
                                                "Field": field.replace('_', ' ').title(),
                                                "Value": val
                                            })
                                
                                if entity_list:
                                    st.markdown("**Entities:**")
                                    st.dataframe(
                                        pd.DataFrame(entity_list),
                                        use_container_width=True,
                                        hide_index=True
                                    )
                else:
                    st.warning("No documents found matching your search")
        else:
            st.info("💡 Enter keywords to search across all processed documents")
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🏢 <strong>AI Intelligent Document Parsing</strong></p>
        <p style='font-size: 0.9rem;'>Powered by Advanced OCR, Multi-Language AI & Cloud Processing</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()