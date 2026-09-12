# SecureCodeRAG: Security Evaluation and Defense Pipeline for Retrieval-Augmented Code Generation

[![Build & Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)](https://github.com/siddhigangan/Final-Year-Project)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-informational)](LICENSE)

## Project Purpose

Retrieval-Augmented Generation (RAG) significantly enhances Code LLMs by fetching relevant code snippets, documentation, and context from external repositories. However, if retrieved knowledge bases contain insecure patterns or adversarial poisoning, the Code LLM may generate vulnerable software. **SecureCodeRAG** is a systematic framework to evaluate security risks associated with retrieval-augmented code generation and to develop robust, automated defense mechanisms.

## Project Objectives

1. **Build a Modular RAG Pipeline for Code Generation**: Construct an extensible, end-to-end framework integrating document ingestion, code chunking, vector embeddings, vector stores, context retrieval, and generation.
2. **Measure Baseline Security Risks**: Benchmark and quantify baseline security vulnerabilities in RAG-assisted code generation under clean knowledge conditions.
3. **Evaluate Security Risks Under Data Poisoning**: Analyze vulnerability rates and impact when external code databases are targeted by adversarial data poisoning or backdoor injection attacks.
4. **Develop and Validate Defense Mechanisms**: Design, implement, and benchmark defense strategies (e.g., AST analysis, security filtering, static analysis integration) to mitigate security risks in RAG-generated code.

---

## Current Status & Completed Functionalities

| Step / Module | Status | Description |
| :--- | :---: | :--- |
| **Step 1: Project Foundation** | ✅ Complete | Package structure, configuration system, logging system, directory setup, foundation unit tests. |
| **Step 2: Ingestion & AST Chunking** | 🔄 Planned | Repository & dataset ingestion, language-specific AST splitting strategies. |
| **Step 3: Embeddings & VectorStore** | 🔄 Planned | Vector embedding generation and indexing using FAISS/ChromaDB. |
| **Step 4: RAG Retrieval & Generation** | 🔄 Planned | Top-K context retrieval, prompt formatting, and Code LLM response generation. |
| **Step 5: Poisoning Attacks** | 🔄 Planned | Data poisoning and backdoor injection attack simulations. |
| **Step 6: Defense & Validation** | 🔄 Planned | Multi-layer AST security filtering, static analysis, and security evaluation. |

---

## High-Level Pipeline Architecture

```text
Knowledge Base (Clean / Poisoned)
              │
              ▼
    Ingestion & AST Chunking
              │
              ▼
   Vector Store & Embeddings
              │
              ▼
      Context Retrieval
              │
              ▼
   Security / Defense Filter
              │
              ▼
      Code LLM Generation
              │
              ▼
   Security Evaluation & Metrics
```

---

## Repository Structure

```text
SecureCodeRAG/
│
├── src/                  # Primary source code organized into modular packages
│   ├── ingestion/        # Document & dataset ingestion handlers
│   ├── chunking/         # Code splitting & AST chunking strategies
│   ├── embeddings/       # Code vector embedding generation
│   ├── vectorstore/      # Vector database indexing & retrieval management
│   ├── retrieval/        # Query-based context retrieval logic
│   ├── generation/       # Code LLM prompt construction & generation
│   ├── poisoning/        # Adversarial context poisoning & attack injection
│   ├── defense/          # Multi-layer security filtering & static analysis
│   ├── evaluation/       # Benchmark execution & security metrics calculation
│   ├── config.py         # Centralized configuration management module
│   └── logger.py         # Standardized logging system
│
├── data/                 # Data storage (git-ignored raw/processed data)
│   ├── clean/            # Clean benchmark repositories & datasets
│   ├── poisoned/         # Adversarial/poisoned context datasets
│   └── processed/        # Embedded vectors & processed chunk caches
│
├── configs/              # Project & pipeline configuration files
│   └── default_config.json
│
├── tests/                # Automated unit and integration tests
│   └── test_foundation.py
│
├── experiments/          # Scripts for empirical evaluations & trials
├── results/              # Experiment metrics, plots, and analysis outputs
├── notebooks/            # Exploratory research and analysis notebooks
├── docs/                 # Project documentation and specifications
│
├── README.md             # Project overview and guide
├── requirements.txt      # Project dependencies
├── .gitignore            # Version control exclusion rules
└── .env.example          # Environment variable template
```

---

## Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.9+ installed on your system.
- Git.

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/siddhigangan/Final-Year-Project.git
cd Final-Year-Project

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and set your configuration variables if required:

```bash
cp .env.example .env
```

---

## Running Tests

Run the foundation test suite to verify module setup, configuration parsing, logging, and project directory structure:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Or run the foundation test module directly:

```bash
python -m tests.test_foundation
```

---

## Configuration & Logging

- **Centralized Configuration**: Settings are managed in `configs/default_config.json` and loaded via `src.config.load_config()`. Configuration parameters can be overridden using environment variables (e.g., `CHUNK_SIZE`, `LOG_LEVEL`, `TOP_K`).
- **Logging System**: Initialized via `src.logger.get_logger()`. Output is logged simultaneously to terminal stdout and persistent log files (`logs/app.log`).

---

## Development Approach

We adhere strictly to an iterative engineering methodology:

> **One step → implement → test → document → verify → push.**

Each phase of development is executed cleanly, verified with automated unit tests, documented, and committed to version control prior to advancing to subsequent modules.
