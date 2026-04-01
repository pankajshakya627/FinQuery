# The Journey: From Prototype to FinQuery 2.0

This document chronicles the developmental evolution of the **FinQuery AI** system—from its origins as a corporate bank-specific prototype to its current state as a premium, high-fidelity RAG application.

## 🚀 Phase 1: The Initial Prototype
The project began as an **HDFC RAG Assistant**, specifically designed to handle "Most Important Terms and Conditions" (MITC) documents for credit cards.

### Initial Challenges:
*   **Table Extraction:** Standard libraries like `PyPDF2` struggled to maintain the structural integrity of complex financial fee tables.
*   **Prompt Specificity:** The initial system was hard-coded for one specific bank, limiting its versatility.
*   **Dependency Conflicts:** Early development was plagued by version mismatches in the `transformers` and `LangChain` ecosystems.

---

## 🛠️ Phase 2: Trial, Error & Resilience
Before reaching the stable **v2.0**, we navigated through several critical technical hurdles and alternative approaches.

### What We Tried & Learned:
1.  **Standard Parsing vs. AI-Vision:** 
    *   *Trial:* We initially used recursive character splitting on raw text.
    *   *Error:* Tables were flattened, losing the relationship between fees and card variants.
    *   *Solution:* Integrated **IBM Docling**, which uses AI to "see" and reconstruct multi-column financial layouts into semantic Markdown.
    
2.  **The "rt_detr_v2" Architecture Error:**
    *   *Conflict:* A version mismatch in the `transformers` library caused the entire application to crash during startup.
    *   *Solution:* Meticulously pinned **`transformers==4.49.0`** and **`numpy==1.26.4`** to restore system stability.

3.  **Uvicorn & Import Mismatches:**
    *   *Internal Crisis:* During a major rebranding refactor, several naming mismatches (e.g., `check_mongodb_connection` vs `check_mongodb_health`) caused `uvicorn` to enter a crash loop.
    *   *Fix:* Mapped all backend routes to a unified connection-handling interface (`MongoDB.disconnect()`, `MongoDB.get_db()`).

---

## 🎨 Phase 3: The FinQuery Rebranding
The move to **FinQuery** wasn't just a name change—it was a complete product pivot toward a premium, platform-agnostic experience.

### Modernization Steps:
*   **Branding Wipe:** Removed all corporate bank references to create a clean, professional "FinQuery AI" identity.
*   **Aesthetic Shift:** Transitioned from a "utility" design to a **Glassmorphic** modern onyx aesthetic.
*   **Theme Engine:** Implemented a native Light/Dark toggle with browser-level persistence (`localStorage`).

---

## 🏆 Current State: FinQuery 2.0
FinQuery is now a robust, high-fidelity RAG engine capable of:
*   Parsing complex financial policy documents with extreme precision.
*   Navigating multi-turn conversations with context memory.
*   Providing real-time system health analytics via a modernized dashboard.

> [!NOTE]
> FinQuery is optimized for use with **Ollama** (local) or **OpenAI** (cloud), providing a versatile deployment path for both privacy-sensitive and performance-first environments.
