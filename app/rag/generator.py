"""
LLM Answer Generation Layer.
Maps to architecture: Online Query Path → LLM Answer Generation
Uses LangChain chains with custom prompts for grounded Q&A.
Supports: OpenAI, Ollama (local), or any OpenAI-compatible API.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document as LCDocument

from app.core.config import get_settings

import structlog

logger = structlog.get_logger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════
#  PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are FinQuery AI, an expert high-fidelity Financial Assistant powered by RAG (Retrieval-Augmented Generation). You specialized in answering questions about complex credit card policies, MITC documents, fees, and rewards with extreme precision.

CRITICAL RULES:
1. Answer ONLY based on the provided technical context from the financial documents.
2. If the context doesn't contain sufficient information, clearly state that
3. Be precise with numbers, fees, percentages, and card names
4. Use ₹ symbol for Indian Rupees (e.g., ₹10,000 not Rs 10000)
5. When multiple card variants have different terms, present them clearly
6. Cite specific sections when possible
7. Keep answers clear, structured, and actionable
8. For fee calculations, show the math step by step
9. Never make up information not present in the context

FORMATTING:
- Use bullet points for lists of charges or card variants
- Bold important numbers and card names
- Group information logically (e.g., Premium cards vs Entry-level)"""

QA_PROMPT_TEMPLATE = """Based on the following extracted context from the credit policy documents:

{context}

---

Question: {input}

Provide a comprehensive, accurate answer. If the context doesn't fully address the question, state what information is available and what's missing."""


# ═══════════════════════════════════════════════════════════════
#  ANSWER GENERATOR
# ═══════════════════════════════════════════════════════════════

class AnswerGenerator:
    """
    Generates grounded answers using LLM + retrieved context.
    Supports: OpenAI, Ollama (local), or any OpenAI-compatible API.
    Provider selected via LLM_PROVIDER in .env (openai | ollama).
    """

    def __init__(self):
        self._llm = None
        self._chain = None

    def initialize(self):
        """Initialize LLM and QA chain based on configured provider."""
        provider = getattr(settings, "llm_provider", "openai").lower()

        logger.info("initializing_llm",
                     provider=provider,
                     model=settings.llm_model,
                     temperature=settings.llm_temperature)

        if provider == "ollama":
            self._llm = self._init_ollama()
        elif provider == "azure":
            self._llm = self._init_azure_openai()
        else:
            self._llm = self._init_openai()

        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(QA_PROMPT_TEMPLATE),
        ])

        # Create stuff documents chain
        self._chain = create_stuff_documents_chain(
            llm=self._llm,
            prompt=prompt,
        )

        logger.info("llm_ready", provider=provider, model=settings.llm_model)

    def _init_openai(self):
        """Initialize OpenAI (or OpenAI-compatible) LLM."""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key,
        )

    def _init_azure_openai(self):
        """Initialize Azure OpenAI LLM."""
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=settings.azure_openai_ad_deployment_name,
            openai_api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    def _init_ollama(self):
        """Initialize Ollama local LLM."""
        from langchain_community.chat_models import ChatOllama
        ollama_base_url = getattr(settings, "ollama_base_url", "http://localhost:11434")
        return ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            base_url=ollama_base_url,
        )

    async def generate(
        self,
        question: str,
        context_docs: list[LCDocument],
        conversation_history: list[dict] = None,
    ) -> str:
        """
        Generate an answer given a question and retrieved context.

        Args:
            question: User's question
            context_docs: Retrieved LangChain Document objects
            conversation_history: Optional previous messages for multi-turn

        Returns:
            Generated answer string
        """
        if not self._chain:
            self.initialize()

        try:
            # Add conversation context if multi-turn
            enhanced_question = question
            if conversation_history:
                history_text = "\n".join(
                    f"{msg['role'].title()}: {msg['content']}"
                    for msg in conversation_history[-4:]  # Last 4 messages
                )
                enhanced_question = f"Previous conversation:\n{history_text}\n\nCurrent question: {question}"

            # Invoke chain
            answer = await self._chain.ainvoke({
                "input": enhanced_question,
                "context": context_docs,
            })

            logger.info("answer_generated",
                        question=question[:80],
                        answer_length=len(answer))

            return answer

        except Exception as e:
            logger.error("generation_failed", error=str(e))
            return self._fallback_answer(question, context_docs, str(e))

    def _fallback_answer(
        self,
        question: str,
        context_docs: list[LCDocument],
        error: str,
    ) -> str:
        """Generate a fallback answer when LLM is unavailable."""
        if not context_docs:
            return (
                "I couldn't find relevant information in the indexed policy documents. "
                "for your question. Please try rephrasing or ask about specific "
                "credit card fees, charges, rewards, or policies."
            )

        # Provide context directly when LLM fails
        context_summary = "\n\n".join(
            f"**Section {i+1}** *(page {doc.metadata.get('page_number', '?')})*:\n{doc.page_content[:400]}"
            for i, doc in enumerate(context_docs[:3])
        )

        return (
            f"⚠️ **LLM unavailable** — showing raw retrieved context instead.\n\n"
            f"*Error: {error}*\n\n"
            f"**Most relevant sections found:**\n\n{context_summary}\n\n"
            f"---\n"
            f"💡 To enable AI answers, either:\n"
            f"- Add OpenAI credits at platform.openai.com/settings/billing\n"
            f"- Or switch to Ollama (free local LLM) — set `LLM_PROVIDER=ollama` in `.env`"
        )

    @property
    def model_name(self) -> str:
        return settings.llm_model


# ── Singleton ──
answer_generator = AnswerGenerator()
