#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web API Interface for RAG Service
Provides HTTP REST API based on Flask.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import traceback
from rag_service import OpenSearchRAGService
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Create Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)
# Global RAG service instance
rag_service = None

def init_rag_service():
    """Initialize the RAG service"""
    global rag_service
    try:
        logger.info("Initializing RAG service...")
        rag_service = OpenSearchRAGService()
        
        # Load and index data
        documents = rag_service.load_cognition_data()
        if documents:
            success = rag_service.index_documents(documents)
            if success:
                logger.info("RAG service initialization successful")
                return True
            else:
                logger.error("Document indexing failed")
                return False
        else:
            logger.error("No documents loaded")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing RAG service: {e}")
        logger.error(traceback.format_exc())
        return False

# Note: The before_first_request decorator has been removed in Flask 2.2+
# The service is now initialized manually in the main function

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {e}")
    logger.error(traceback.format_exc())
    return jsonify({
        "error": "Internal Server Error",
        "message": str(e)
    }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if rag_service is None:
        return jsonify({
            "status": "error",
            "message": "RAG service not initialized"
        }), 503
    
    try:
        stats = rag_service.get_stats()
        return jsonify({
            "status": "healthy",
            "service": "RAG API",
            "stats": stats
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 503

@app.route('/search', methods=['POST'])
def search_patterns():
    """Search for similar experiment trigger patterns"""
    if rag_service is None:
        return jsonify({
            "error": "RAG service not initialized"
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "Request body cannot be empty"
            }), 400
        
        query = data.get('query', '').strip()
        if not query:
            return jsonify({
                "error": "Query parameter cannot be empty"
            }), 400
        
        k = data.get('k', 5)
        similarity_threshold = data.get('similarity_threshold', 0.6)
        
        # Validate parameters
        if not isinstance(k, int) or k <= 0 or k > 50:
            return jsonify({
                "error": "Parameter k must be an integer between 1 and 50"
            }), 400
        
        if not isinstance(similarity_threshold, (int, float)) or similarity_threshold < 0 or similarity_threshold > 1:
            return jsonify({
                "error": "Similarity threshold must be a value between 0 and 1"
            }), 400
        
        # Perform the search
        results = rag_service.search_similar_patterns(
            query=query,
            k=k,
            similarity_threshold=similarity_threshold
        )
        
        return jsonify({
            "query": query,
            "total_results": len(results),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error during search: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Search failed",
            "message": str(e)
        }), 500

@app.route('/paper/<paper_key>', methods=['GET'])
def get_paper_documents(paper_key):
    """Get all related documents based on the paper key"""
    if rag_service is None:
        return jsonify({
            "error": "RAG service not initialized"
        }), 503
    
    try:
        if not paper_key.strip():
            return jsonify({
                "error": "Paper key cannot be empty"
            }), 400
        
        results = rag_service.get_document_by_paper(paper_key)
        
        return jsonify({
            "paper_key": paper_key,
            "total_documents": len(results),
            "documents": results
        })
        
    except Exception as e:
        logger.error(f"Error during paper key search: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Search failed",
            "message": str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_statistics():
    """Get index statistics"""
    if rag_service is None:
        return jsonify({
            "error": "RAG service not initialized"
        }), 503
    
    try:
        stats = rag_service.get_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to get statistics",
            "message": str(e)
        }), 500

@app.route('/reinit', methods=['POST'])
def reinitialize_service():
    """Re-initialize the RAG service"""
    try:
        success = init_rag_service()
        if success:
            return jsonify({
                "status": "success",
                "message": "RAG service re-initialized successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "RAG service re-initialization failed"
            }), 500
            
    except Exception as e:
        logger.error(f"Error during re-initialization: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/', methods=['GET'])
def api_info():
    """API information and usage instructions"""
    return jsonify({
        "service": "RAG API for Cognition Database",
        "description": "RAG service based on OpenSearch for searching and retrieving experimental trigger patterns",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /stats": "Get index statistics",
            "POST /search": "Search for similar experiment trigger patterns",
            "GET /paper/<paper_key>": "Get documents by paper key",
            "POST /reinit": "Re-initialize service"
        },
        "search_example": {
            "url": "/search",
            "method": "POST",
            "body": {
                "query": "The model performs poorly on long sequences",
                "k": 5,
                "similarity_threshold": 0.6
            }
        }
    })

if __name__ == '__main__':
    print("Starting RAG API service...")
    print("Initializing RAG Service...")
    
    # Initialize the RAG service before starting the Flask application
    success = init_rag_service()
    if not success:
        print("❌ RAG service initialization failed, please check the logs")
        exit(1)
    
    print("✅ RAG service initialized successfully")
    print("API Documentation: http://localhost:5000/")
    print("Health Check: http://localhost:5000/health")
    print("Statistics: http://localhost:5000/stats")
    
    app.run(
        host='0.0.0.0',
        port=13142,
        debug=False,
        threaded=True
    )