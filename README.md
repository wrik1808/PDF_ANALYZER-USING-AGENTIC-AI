# 🤖 Agentic AI PDF Analyzer

> An intelligent multi-agent AI system that analyzes PDF documents, extracts meaningful information, generates concise summaries, and produces actionable insights using LLM-powered agents.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react)
![LangChain](https://img.shields.io/badge/LangChain-AI%20Framework-1C3C3C?style=for-the-badge)
![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Overview

**Agentic AI PDF Analyzer** is an AI-powered document analysis application designed to automatically understand and process PDF documents.

Instead of relying on a single AI prompt, the system uses a **multi-agent architecture**, where different AI agents are responsible for different stages of document understanding.

The system can:

- 📄 Upload and process PDF documents
- 🔍 Classify document types
- 🧠 Extract important information
- 📝 Generate concise summaries
- 💡 Generate useful insights
- ⚠️ Identify potential risks or missing information
- ❓ Generate follow-up questions
- 📊 Present analysis results through a web interface

The project combines **Agentic AI, Large Language Models, LangChain, FastAPI, React, and PDF processing** into a complete full-stack AI application.

---

# ✨ Features

## 📄 PDF Processing

Upload a PDF document and automatically extract its text for further analysis.

Supported documents can include:

- Research Papers
- Technical Reports
- Contracts
- Legal Documents
- Resumes
- Invoices
- Notes
- Other textual documents

---

## 🤖 Multi-Agent AI Architecture

The application uses multiple specialized AI agents.

### 1. 🔍 Document Classifier Agent

Identifies the type of uploaded document.

Example:

```text
Input:
Research paper PDF

Output:
Research Paper
