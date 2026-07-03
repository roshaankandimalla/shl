from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DATA_DIR = DATA_DIR / "chroma"

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
CATALOG_PATH = RAW_DATA_DIR / "shl_product_catalog.json"
ASSESSMENT_DOCUMENTS_PATH = PROCESSED_DATA_DIR / "assessment_documents.json"
SPLADE_VECTORS_PATH = PROCESSED_DATA_DIR / "splade_sparse_vectors.json"
QUERY_EMBEDDINGS_CACHE_PATH = PROCESSED_DATA_DIR / "query_embeddings_cache.json"
CHROMA_COLLECTION_NAME = "shl_assessments"

DENSE_TOP_K = 30
SPARSE_TOP_K = 30
RRF_TOP_K = 30
RERANK_TOP_K = 10
RRF_K = 60

DENSE_EMBED_MODEL = "voyage-4-large"
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"
RERANK_MODEL = "rerank-2.5"

GENERATION_MODEL = "gemini-3.1-flash-lite"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
