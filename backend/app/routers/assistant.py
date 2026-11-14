"""
BoxOfficeIQ - AI Assistant Router
Hybrid RAG + Data Query Assistant combining safe pandas aggregations (CinemaDataTool),
vector knowledge base retrieval (RAGPipeline), and LLM generation (with offline intelligent fallback).
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.assistant.sql_tool import CinemaDataTool
from app.assistant.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assistant", tags=["assistant"])

_sql_tool: Optional[CinemaDataTool] = None
_rag_pipeline: Optional[RAGPipeline] = None


def get_sql_tool() -> CinemaDataTool:
    global _sql_tool
    if _sql_tool is None:
        _sql_tool = CinemaDataTool()
    return _sql_tool


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query message")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=[], description="Previous chat turns")


SUGGESTED_QUESTIONS = [
    "Which cinema venue recorded the highest attendance?",
    "What is our best-performing forecasting model and what is its MAE?",
    "How does weekend attendance compare to weekday attendance?",
    "What are the top features driving cinema attendance according to SHAP?",
    "What is the projected 7-day demand for PVR Cinemas Juhu?",
    "How does DistilBERT sentiment analysis score recent movie releases?",
    "Which cinema locations currently have the lowest average occupancy?",
    "How does the what-if simulation calculate staffing and concessions?",
]


@router.get("/suggested-questions")
def get_suggested_questions() -> List[str]:
    """Returns curated starter prompts for the AI decision support assistant."""
    return SUGGESTED_QUESTIONS


@router.post("/chat")
def chat_with_assistant(req: ChatRequest) -> Dict[str, Any]:
    """
    Hybrid decision support endpoint.
    Determines query classification (Data SQL / Documentation RAG / Hybrid),
    retrieves context, and synthesizes answers via OpenAI or built-in contextual synthesis.
    """
    query = req.message.strip()
    if not query:
        return {
            "response": "Please enter a question regarding cinema attendance, forecasts, model metrics, or methodology.",
            "sources": [],
            "query_type": "fallback",
            "data_used": {},
        }

    sql_tool = get_sql_tool()
    rag_pipeline = get_rag_pipeline()

    q_lower = query.lower()

    # Determine query classification
    is_doc_query = any(w in q_lower for w in [
        "architecture", "methodology", "what is shap", "how does", "pipeline", "tech stack",
        "doc", "guide", "explain shap", "distilbert", "how was", "leakage", "data dictionary"
    ])
    is_data_query = any(w in q_lower for w in [
        "highest", "lowest", "average", "top", "rank", "attendance", "how many",
        "forecast", "mae", "rmse", "wape", "best model", "theatre", "occupancy", "trend"
    ])

    query_type = "hybrid" if (is_doc_query and is_data_query) else ("rag" if is_doc_query else "sql")

    # 1. Gather SQL/Data context
    sql_result = sql_tool.execute_query(query)
    data_used = sql_result.get("data", {})

    # 2. Gather RAG document chunks
    rag_chunks = rag_pipeline.retrieve(query, k=3)
    sources = list({c["source"] for c in rag_chunks})

    # 3. Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if openai_api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_api_key)

            context_snippets = "\n\n".join([f"[{c['source']}]: {c['content']}" for c in rag_chunks])
            system_prompt = (
                "You are BoxOfficeIQ AI Assistant, an expert cinema demand forecasting consultant.\n"
                "Answer the user's question clearly, grounded in the provided factual documentation and data.\n"
                f"Data Context:\n{sql_result.get('text', '')}\n\n"
                f"Documentation Context:\n{context_snippets}\n"
            )

            messages = [{"role": "system", "content": system_prompt}]
            for turn in (req.conversation_history or []):
                messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
            messages.append({"role": "user", "content": query})

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
            )
            llm_text = completion.choices[0].message.content
            return {
                "response": llm_text,
                "answer": llm_text,
                "sources": sources,
                "citations": sources,
                "query_type": query_type,
                "data_used": data_used,
            }
        except Exception as e:
            logger.warning(f"OpenAI call failed ({e}), falling back to offline synthesizer.")

    # 4. Contextual Offline Synthesizer
    answer_parts = []

    if query_type in ("sql", "hybrid"):
        answer_parts.append(sql_result.get("text", ""))

    if query_type in ("rag", "hybrid") and rag_chunks:
        doc_summary = f"According to {rag_chunks[0]['source']}: {rag_chunks[0]['content']}"
        answer_parts.append(doc_summary)

    if not answer_parts:
        answer_parts.append(sql_result.get("text", "BoxOfficeIQ platform analytics are available across all 15 cinema venues."))

    final_response = "\n\n".join(answer_parts)

    return {
        "response": final_response,
        "answer": final_response,
        "sources": sources,
        "citations": sources,
        "query_type": query_type,
        "data_used": data_used,
    }
