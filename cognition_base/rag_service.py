#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Service based on OpenSearch
For indexing and querying EXPERIMENTAL_TRIGGER_PATTERNS in the cognition database.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGDocument:
    """RAG Document Data Structure"""
    id: str
    paper_key: str  # JSON filename
    design_insight: str
    experimental_trigger_patterns: str
    background: str
    algorithmic_innovation: str
    implementation_guidance: str
    design_ai_instructions: str
    embedding: Optional[List[float]] = None

class LocalEmbeddingClient:
    """Local SentenceTransformer Embedding Client"""
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        """
        Initializes the local embedding client
        
        Args:
            model_name: SentenceTransformer model name
        """
        logger.info(f"Loading local embedding model: {model_name}")
        
        # Force the use of CPU
        self.model = SentenceTransformer(model_name, device='cpu')
        self.model_name = model_name
        
        logger.info(f"Successfully loaded model {model_name} (using CPU)")
        
        # Get the embedding dimension of the model
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding vector dimension: {self.embedding_dim}")
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Gets the embedding vectors for the texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # For e5 models, add a special prefix
            processed_texts = [f"query: {text}" if not text.startswith("query:") else text for text in texts]
            
            # Use model encoding for text, process using batches to save memory
            all_embeddings = []
            
            for i in range(0, len(processed_texts), batch_size):
                batch_texts = processed_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=min(batch_size, len(batch_texts))
                )
                all_embeddings.extend(batch_embeddings.tolist())
                
                logger.debug(f"Processed {min(i + batch_size, len(processed_texts))}/{len(processed_texts)} texts")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.embedding_dim] * len(texts)
    
    def get_single_embedding(self, text: str) -> List[float]:
        """
        Gets the embedding vector for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # For e5 models, add query prefix
            processed_text = f"query: {text}" if not text.startswith("query:") else text
            
            embedding = self.model.encode(
                processed_text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error getting single embedding: {e}")
            return [0.0] * self.embedding_dim

class OpenSearchRAGService:
    """OpenSearch-based RAG Service"""
    
    def __init__(
        self,
        opensearch_host: str = "localhost",
        opensearch_port: int = 9200,
        index_name: str = "cognition_rag",
        model_name: str = "intfloat/e5-base-v2"
    ):
        """
        Initializes the RAG service
        
        Args:
            opensearch_host: OpenSearch host address
            opensearch_port: OpenSearch port
            index_name: Index name
            model_name: Local embedding model name
        """
        self.index_name = index_name
        
        # Initialize OpenSearch client
        self.client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        
        # Initialize local embedding client
        logger.info("Initializing local embedding client")
        self.embedding_client = LocalEmbeddingClient(model_name=model_name)
        
        # Create the index
        self._create_index()
    
    def _create_index(self):
        """Creates the OpenSearch index"""
        # Use the actual dimension of the embedding client
        embedding_dim = self.embedding_client.embedding_dim
        
        # Improved index configuration, explicitly enables KNN
        index_body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "knn": True,  # Explicitly enable KNN
                    "knn.algo_param.ef_search": 100  # Set the KNN search parameter
                },
                "analysis": {
                    "analyzer": {
                        "chinese_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "paper_key": {
                        "type": "keyword"
                    },
                    "design_insight": {
                        "type": "text",
                        "analyzer": "chinese_analyzer"
                    },
                    "experimental_trigger_patterns": {
                        "type": "text",
                        "analyzer": "chinese_analyzer"
                    },
                    "background": {
                        "type": "text",
                        "analyzer": "chinese_analyzer"
                    },
                    "algorithmic_innovation": {
                        "type": "text",
                        "analyzer": "chinese_analyzer"
                    },
                    "implementation_guidance": {
                        "type": "text",
                        "analyzer": "chinese_analyzer"
                    },
                    "design_ai_instructions": {
                        "type": "text",
                        "analyzer": "chinese_analyzer"
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dim,  # Use the actual embedding dimension
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib", # Use the nmslib engine
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    }
                }
            }
        }
        
        # Delete existing index (if it exists)
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            logger.info(f"Deleted existing index: {self.index_name}")
        
        # Create new index
        self.client.indices.create(index=self.index_name, body=index_body)
        logger.info(f"Created index: {self.index_name} (embedding dimension: {embedding_dim})")
    
    def _generate_document_id(self, paper_key: str, design_insight: str) -> str:
        """Generates the document ID"""
        content = f"{paper_key}_{design_insight}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _extract_filename(self, filepath: str) -> str:
        """Extract file name (without extension) from file path"""
        return Path(filepath).stem
    
    def load_cognition_data(self, data_dir: str = "cognition") -> List[RAGDocument]:
        """
        Loads all JSON files in the cognition data directory
        
        Args:
            data_dir: Path to the data directory
            
        Returns:
            List of RAGDocument objects
        """
        documents = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            return documents
        
        # Traverses all JSON files
        for json_file in data_path.glob("*.json"):
            try:
                logger.info(f"Processing file: {json_file}")
                paper_key = self._extract_filename(str(json_file))
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Processes each design insight
                for item in data:
                    if isinstance(item, dict) and "EXPERIMENTAL_TRIGGER_PATTERNS" in item:
                        # Ensure that EXPERIMENTAL_TRIGGER_PATTERNS is not empty
                        trigger_patterns = item.get("EXPERIMENTAL_TRIGGER_PATTERNS", "").strip()
                        if not trigger_patterns:
                            logger.warning(f"Skipping empty EXPERIMENTAL_TRIGGER_PATTERNS, file: {paper_key}")
                            continue
                            
                        doc_id = self._generate_document_id(paper_key, item.get("DESIGN_INSIGHT", ""))
                        
                        doc = RAGDocument(
                            id=doc_id,
                            paper_key=paper_key,
                            design_insight=item.get("DESIGN_INSIGHT", ""),
                            experimental_trigger_patterns=trigger_patterns,
                            background=item.get("BACKGROUND", ""),
                            algorithmic_innovation=item.get("ALGORITHMIC_INNOVATION", ""),
                            implementation_guidance=item.get("IMPLEMENTATION_GUIDANCE", ""),
                            design_ai_instructions=item.get("DESIGN_AI_INSTRUCTIONS", "")
                        )
                        
                        documents.append(doc)
                        
            except Exception as e:
                logger.error(f"Error processing file {json_file}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Generate embedding vectors in batches - only based on EXPERIMENTAL_TRIGGER_PATTERNS
        if documents:
            logger.info("Starting to generate embedding vectors (based on EXPERIMENTAL_TRIGGER_PATTERNS)...")
            texts = [doc.experimental_trigger_patterns for doc in documents]
            embeddings = self.embedding_client.get_embeddings(texts)
            
            # Assign embedding vectors to corresponding documents
            for i, doc in enumerate(documents):
                if i < len(embeddings):
                    doc.embedding = embeddings[i]
                else:
                    doc.embedding = [0.0] * self.embedding_client.embedding_dim
            
            logger.info(f"Successfully generated {len(embeddings)} embedding vectors")
        
        return documents
    
    def index_documents(self, documents: List[RAGDocument]) -> bool:
        """
        Indexes documents into OpenSearch
        
        Args:
            documents: List of documents to index
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            for doc in documents:
                doc_body = {
                    "paper_key": doc.paper_key,
                    "design_insight": doc.design_insight,
                    "experimental_trigger_patterns": doc.experimental_trigger_patterns,
                    "background": doc.background,
                    "algorithmic_innovation": doc.algorithmic_innovation,
                    "implementation_guidance": doc.implementation_guidance,
                    "design_ai_instructions": doc.design_ai_instructions,
                    "embedding": doc.embedding
                }
                
                self.client.index(
                    index=self.index_name,
                    id=doc.id,
                    body=doc_body
                )
            
            # Refresh the index to ensure documents are searchable
            self.client.indices.refresh(index=self.index_name)
            logger.info(f"Successfully indexed {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return False
    
    def search_similar_patterns(
        self,
        query: str,
        k: int = 5,
        similarity_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Searches for similar experimental trigger patterns based on the query text.
        
        Args:
            query: The query text.
            k: The number of results to return.
            similarity_threshold: The similarity threshold.
            
        Returns:
            A list of matching documents, sorted by similarity.
        """
        # Generate a query embedding
        query_embedding = self.embedding_client.get_single_embedding(query)
        logger.info(f"Query embedding dimension: {len(query_embedding)}")
        
        # First, try to use the script_score query (more compatible approach)
        search_body = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            },
            "_source": {
                "excludes": ["embedding"]  # Exclude embedding vectors to reduce response size
            }
        }
        
        try:
            response = self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            logger.info(f"OpenSearch returned {len(response['hits']['hits'])} results")
            
            results = []
            for hit in response['hits']['hits']:
                score = hit['_score']
                # Normalize script_score's score to the range of 0-1 (subtract 1.0 because we added 1.0 previously)
                normalized_score = max(0.0, score - 1.0)
                
                logger.debug(f"Document {hit['_id']} original score: {score}, normalized score: {normalized_score}")
                
                # Apply similarity threshold filtering
                if normalized_score >= similarity_threshold:
                    # Maintain the original simple data structure
                    result = {
                        "id": hit['_id'],
                        "score": normalized_score,
                        "paper_key": hit['_source']['paper_key'],
                        "DESIGN_INSIGHT": hit['_source']['design_insight'],
                        "EXPERIMENTAL_TRIGGER_PATTERNS": hit['_source']['experimental_trigger_patterns'],  # Primary matching field
                        "BACKGROUND": hit['_source']['background'],
                        "ALGORITHMIC_INNOVATION": hit['_source']['algorithmic_innovation'],
                        "IMPLEMENTATION_GUIDANCE": hit['_source']['implementation_guidance'],
                        "DESIGN_AI_INSTRUCTIONS": hit['_source']['design_ai_instructions']
                    }
                    results.append(result)
            
            logger.info(f"Query '{query}' returned {len(results)} results after threshold filtering")
            return results
            
        except Exception as e:
            logger.error(f"script_score search failed, attempting KNN search: {e}")
            
            # If script_score fails, attempt the KNN search
            return self._knn_search_fallback(query, query_embedding, k, similarity_threshold)
    
    def _knn_search_fallback(self, query: str, query_embedding: List[float], k: int, similarity_threshold: float) -> List[Dict[str, Any]]:
        """KNN Search Fallback Method"""
        search_body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": k
                    }
                }
            },
            "_source": {
                "excludes": ["embedding"]
            }
        }
        
        try:
            response = self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            results = []
            for hit in response['hits']['hits']:
                score = hit['_score']
                
                # Apply similarity threshold filtering
                if score >= similarity_threshold:  # KNN scores already range from 0-1
                    # Maintain the original simple data structure
                    result = {
                        "id": hit['_id'],
                        "score": score,
                        "paper_key": hit['_source']['paper_key'],
                        "DESIGN_INSIGHT": hit['_source']['design_insight'],
                        "EXPERIMENTAL_TRIGGER_PATTERNS": hit['_source']['experimental_trigger_patterns'],  # Primary Matching Field
                        "BACKGROUND": hit['_source']['background'],
                        "ALGORITHMIC_INNOVATION": hit['_source']['algorithmic_innovation'],
                        "IMPLEMENTATION_GUIDANCE": hit['_source']['implementation_guidance'],
                        "DESIGN_AI_INSTRUCTIONS": hit['_source']['design_ai_instructions']
                    }
                    results.append(result)
            
            logger.info(f"KNN query '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"KNN search also failed: {e}")
            return []
    
    def get_document_by_paper(self, paper_key: str) -> List[Dict[str, Any]]:
        """
        Gets all related documents by paper key
        
        Args:
            paper_key: Paper key (filename)
            
        Returns:
            A list of matching documents.
        """
        search_body = {
            "query": {
                "term": {
                    "paper_key": paper_key
                }
            },
            "_source": {
                "excludes": ["embedding"]
            }
        }
        
        try:
            response = self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            results = []
            for hit in response['hits']['hits']:
                # Maintain the original simple data structure
                result = {
                    "id": hit['_id'],
                    "paper_key": hit['_source']['paper_key'],
                    "DESIGN_INSIGHT": hit['_source']['design_insight'],
                    "EXPERIMENTAL_TRIGGER_PATTERNS": hit['_source']['experimental_trigger_patterns'],
                    "BACKGROUND": hit['_source']['background'],
                    "ALGORITHMIC_INNOVATION": hit['_source']['algorithmic_innovation'],
                    "IMPLEMENTATION_GUIDANCE": hit['_source']['implementation_guidance'],
                    "DESIGN_AI_INSTRUCTIONS": hit['_source']['design_ai_instructions']
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during paper key search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.client.indices.stats(index=self.index_name)
            doc_count = stats['indices'][self.index_name]['total']['docs']['count']
            
            # Get the number of distinct papers
            aggs_body = {
                "size": 0,
                "aggs": {
                    "unique_papers": {
                        "cardinality": {
                            "field": "paper_key"
                        }
                    }
                }
            }
            
            aggs_response = self.client.search(
                index=self.index_name,
                body=aggs_body
            )
            
            unique_papers = aggs_response['aggregations']['unique_papers']['value']
            
            return {
                "total_documents": doc_count,
                "unique_papers": unique_papers,
                "index_name": self.index_name
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

def main():
    """Main function: demonstrates the usage of the RAG service"""
    # Initialize the RAG service
    rag_service = OpenSearchRAGService()
    
    # Load the data
    documents = rag_service.load_cognition_data()
    
    if not documents:
        logger.error("No documents loaded")
        return
    
    # Index the documents
    success = rag_service.index_documents(documents)
    
    if not success:
        logger.error("Document indexing failed")
        return
    
    # Print statistics
    stats = rag_service.get_stats()
    print(f"\n=== Index Statistics ===")
    print(f"Total Documents: {stats.get('total_documents', 0)}")
    print(f"Number of Unique Papers: {stats.get('unique_papers', 0)}")
    print(f"Index Name: {stats.get('index_name', 'unknown')}")
    
    # Example queries
    test_queries = [
        "The model performs poorly on long sequences",
        "Attention mechanism computational complexity is too high",
        "The model training is slow",
        "Sequence modeling bottlenecks"
    ]
    
    print(f"\n=== Example Query Results ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = rag_service.search_similar_patterns(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Paper: {result['paper_key']}")
            print(f"     Similarity: {result['score']:.3f}")
            print(f"     Trigger Patterns: {result['EXPERIMENTAL_TRIGGER_PATTERNS'][:100]}...")
            print()

if __name__ == "__main__":
    main()