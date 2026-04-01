# Design System: FinQuery Modern Onyx

FinQuery AI features a custom-built design system, "Modern Onyx," crafted to deliver a sophisticated, minimal, and premium AI experience. This document details the aesthetic logic, typography, and theme engine.

## 🎨 Aesthetic Logic: Glassmorphism

The FinQuery interface uses a "glassmorphic" approach, blending semi-transparent layers with deep blurs to create depth and focus.

### Key Principles:
*   **Translucency:** Uses `rgba` backgrounds with `backdrop-filter: blur(20px)` to allow soft background hints.
*   **Layering:** Elements like the sidebar and the input dock appear to "float" above the application shell.
*   **Borders:** Subtle `1px` borders using high-contrast colors (`rgba(255,255,255,0.1)`) create sharp, premium definition.

---

## 🌓 Theme Engine: Universal Preference

FinQuery supports a native Light and Dark mode, controllable via a persistent theme toggle in the sidebar.

### Theme Variables (`styles.css`):
Themes are managed using CSS custom properties (variables) defined on the `:root` and `[data-theme='dark']` selectors.

| Variable | Light Theme (default) | Dark Theme (Modern Onyx) |
| :--- | :--- | :--- |
| `--bg-app` | `#f8faff` (Soft Ice) | `#0a0c10` (Deep Space) |
| `--bg-card` | `#ffffff` (Pure White) | `rgba(20, 24, 33, 0.7)` (Glass) |
| `--text-main` | `#1e293b` (Slate) | `#f1f5f9` (Cloud) |
| `--accent` | `#336dec` (Electric Blue) | `#4f8aff` (Vibrant Blue) |
| `--border` | `#e2e8f0` (Thin Gray) | `rgba(255, 255, 255, 0.08)` (Soft Line) |

### Persistence Logic (`app.js`):
The user's theme choice is automatically synchronized with their browser's **localStorage**, ensuring their preference is remembered on subsequent visits.

---

## 🖋️ Typography: The Outfit Fontset

FinQuery uses the **Outfit** font family (Google Fonts) for its balance of geometric precision and modern readability.

*   **Headings:** `700` or `800` weight for bold, architectural clarity and a strong brand presence.
*   **Body Text:** `400` or `500` weight with increased line-height (`1.6`) for reduced eye-strain in dense financial documents.
*   **Code:** **JetBrains Mono** for technical sections, retrieval stats, and reasoning logs.

---

## ✨ Micro-Animations & Interactivity

Fluid motion is core to the FinQuery experience, providing real-time feedback for AI operations.

### Key Animations:
1.  **Staggered Entrance:** Message bubbles use a native CSS `@keyframes` slide-up animation with a `0.1s` staggered delay for a "natural" conversation feel.
2.  **Magnetic Hover:** Sidebar navigation items feature a smooth `cubic-bezier` scale and color transition when hovered.
3.  **Pipeline Pulse:** Active RAG steps (Embedding, Retrieval, etc.) use a subtle glow pulse to visualize backend processing.
4.  **Glass Transitions:** Theme switching uses a CSS `transition: all 0.4s ease` to softly shift between Light and Dark palettes.

---

## 📱 Responsiveness

The "Modern Onyx" system is built on a **Mobile-First** CSS architecture, ensuring the AI assistant remains usable and beautiful on tablets and mobile devices via fluid grid layouts and flexible flex-containers.

> [!TIP]
> The design system is entirely contained within `app/static/css/styles.css`. No external CSS frameworks (like Tailwind or Bootstrap) were used, allowing for maximum performance and zero dependency overhead.
