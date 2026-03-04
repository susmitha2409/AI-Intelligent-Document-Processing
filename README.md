# AI Intelligent Document Processing Platform

AI-powered multilingual document intelligence platform built using **Surya OCR**, **LLM-based entity extraction**, and **MongoDB search**.

🏆 **Winner – India AI for All Challenge 2026**

Presented at **India AI Summit 2026 – New Delhi**

---

## Key Features

• Advanced OCR using Surya OCR  
• Handles faded and noisy government documents  
• Multilingual text detection and translation  
• AI-powered entity extraction using LLM (Groq + Llama 3)  
• Document search using MongoDB  
• Multi-page navigation with bounding boxes  
• Batch document processing  

---

## System Architecture

![Architecture](docs/architecture.png)

---

## Technology Stack

| Component | Technology |
|--------|--------|
Frontend | Streamlit |
OCR Engine | Surya OCR |
LLM Analysis | Groq Llama 3 |
Translation | Google Translator API |
Database | MongoDB |
Language Detection | Langdetect |

---

## Project Workflow

1️⃣ Upload documents (PDF / Images)

2️⃣ OCR Processing using Surya

3️⃣ Text Translation to English

4️⃣ Entity Extraction using LLM

5️⃣ Store results in MongoDB

6️⃣ Search documents using keywords

---

## Installation

Clone repository

```bash
git clone https://github.com/yourusername/ai-intelligent-document-processing.git
cd ai-intelligent-document-processing
