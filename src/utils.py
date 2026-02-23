# src/utils.py
import re
from loguru import logger
from langchain_core.output_parsers import BaseOutputParser

def prune_legal_context(context, max_chars: int = 3500) -> str:
    """
    Safely joins and prunes legal context to stay within token limits
    while preserving the integrity of legal breadcrumbs [[LAW: ...]].
    """
    # 1. Handle list input from LangGraph state
    if isinstance(context, list):
        context = "\n\n".join([str(c) for c in context])
    
    if not isinstance(context, str):
        context = str(context)

    # 2. Aggressive whitespace cleanup to save token space
    context = re.sub(r'\s+', ' ', context).strip()
    
    if len(context) <= max_chars:
        return context
    
    # 3. Smart Pruning: Look for the last tag start [[LAW:
    pruned = context[:max_chars]
    last_tag_index = pruned.rfind('passage: [LAW:')
    
    # If a tag exists, cut right before it so we don't send a partial citation
    if last_tag_index > 0:
        return pruned[:last_tag_index].strip()
        
    return pruned.strip()

def clean_feedback(text: str) -> str:
    """Removes conversational filler and ensures the sentence isn't truncated mid-way."""
    # Remove common prefixes like 'REASON:', 'The response is...', etc.
    text = re.sub(r'^(The response|Evaluation|REASON|Feedback|Score)[:\s]*', '', text, flags=re.IGNORECASE)
    
    # Split into sentences and keep the first two
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean = " ".join(sentences[:2])
    
    # If the text ends abruptly without punctuation, find the last valid sentence end
    if not clean.endswith(('.', '!', '?')):
        last_punct = max(clean.rfind('.'), clean.rfind('!'), clean.rfind('?'))
        if last_punct > 0:
            clean = clean[:last_punct + 1]
        else:
            clean = clean[:147] + "..." # Fallback for very long single sentences
            
    return clean.strip()

class StrictLegalQueryParser(BaseOutputParser[list[str]]):
    """Surgically removes noise and prevents Criminal Domain 'poisoning' of Civil queries."""
    
    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        raw_clean_queries = []
        
        # 1. Standard Cleaning (Numbering, Preamble, Quotes)
        for line in lines:
            line = re.sub(r"^(?:\d+\.?|[-*•]|Query \d+:|\*\*Query.*?\*\*|As a .*?:|Here are .*?:)\s*", "", line.strip())
            line = line.replace('"', '').replace("'", "").replace("+", " ").replace("*", "").strip()
            
            if len(line) > 10 and not line.endswith(":"):
                raw_clean_queries.append(line)

        # 2. DOMAIN SANITIZER: Prevent 'BNS' keywords from leaking into Civil/Family/Property searches
        final_queries = []
        civil_keywords = [
            "sale deed", "registration", "marriage", "divorce", "apartment", "flat", 
            "notarized", "agreement", "tenant", "cpc", "civil", "it act", "cyber", 
            "pan card", "whatsapp fraud", "inheritance", "succession", "will", 
            "father", "son", "share", "constitution", "fundamental rights", 
            "ndps", "drugs", "narcotics", "article" 
        ]
        criminal_noise = [
            "under bns", "under bnss", "in bns", "in bnss", "ipc equivalent", 
            "punishment and bailability", "corresponding bns", "bns section"
        ]

        for query in raw_clean_queries:
            # Check if this specific query is about a Civil/Family/Property matter
            if any(key in query.lower() for key in civil_keywords):
                # Strip out criminal keywords that cause 'Zero Results' in the vector store
                for noise in criminal_noise:
                    query = re.sub(re.escape(noise), "", query, flags=re.IGNORECASE)
                
                # Clean up double spaces created by the regex removal
                query = re.sub(r'\s+', ' ', query).strip()
            
            final_queries.append(query)
        
        # Ensure we don't return 3 identical queries if the cleaner stripped them down
        unique_queries = list(dict.fromkeys(final_queries))

        output = unique_queries[:3]
        logger.info(f"🧹 MMR Cleaned & Sanitized Queries: {output}")
        return output