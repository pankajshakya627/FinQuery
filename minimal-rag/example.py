#!/usr/bin/env python3
"""
Minimal RAG Example Usage
"""
from rag_module.rag import MinimalRAG


def main():
    print("Starting Minimal RAG System...")
    rag = MinimalRAG()

    # Example document
    sample_text = """
    Kilo is an AI code assistant designed for software engineers.
    It helps with debugging, refactoring, writing tests, and understanding codebases.
    Kilo runs locally on your machine and works with your existing projects.
    It supports multiple programming languages including Python, JavaScript, TypeScript, and Go.
    The assistant uses advanced language models to provide accurate code suggestions.
    """

    print("\nAdding document to vector store...")
    chunks_added = rag.add_document(sample_text, {"source": "kilo_overview"})
    print(f"Added {chunks_added} chunks")

    print("\nRunning queries:")

    questions = [
        "What is Kilo?",
        "What programming languages does Kilo support?",
        "Where does Kilo run?",
        "What can Kilo help with?"
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer = rag.query(q)
        print(f"Answer: {answer.strip()}")


if __name__ == "__main__":
    main()
