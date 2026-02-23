import sys
import time
from loguru import logger
from psycopg_pool import ConnectionPool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

# Internal Imports
from .config import DB_URI
from .agent import create_graph
from .engine import get_vector_store
from .processor import run_ingestion_pipeline

def connect_with_retry(uri, kwargs, retries=3, delay=5):
    """Attempt to establish a connection pool with retries for Neon cold-starts."""
    for attempt in range(retries):
        try:
            pool = ConnectionPool(conninfo=uri, max_size=10, kwargs=kwargs)
            # Pre-ping the database to ensure SSL handshake is complete
            with pool.connection() as conn:
                conn.execute("SELECT 1")
            return pool
        except Exception as e:
            logger.warning(f"⚠️  Postgres connection attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

def main():
    logger.info("⚖️  Legal Advisor System Starting Up...")

    try:
        # 1. Initialize Vector Store connection
        vs = get_vector_store()
        client = vs.client
        collection_name = "indian_legal_library"

        # 2. Configure the Postgres Connection Pool
        # Optimized for Neon Serverless
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": None,
            "sslmode": "require",
            "channel_binding" : "require",
            "tcp_user_timeout": 10000 # 10 seconds
        }

        # 3. Production Gate: Load Once Logic
        try:
            collection_info = client.get_collection(collection_name)
            point_count = collection_info.points_count
            if point_count > 0:
                logger.info(f"✅ Production Index Ready: Found {point_count} legal snippets.")
            else:
                logger.warning(f"⚠️  Vector Store '{collection_name}' is empty.")
                run_ingestion_pipeline(vs)
        except Exception as e:
            logger.info("📥 Initializing first-time setup and indexing...")
            run_ingestion_pipeline(vs)

        # 4. Establish Postgres Connection Pool with Retry
        pool = connect_with_retry(DB_URI, connection_kwargs)
        
        with pool:
            # 5. Initialize Postgres Checkpointer
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            logger.info("🗄️  Checkpointer linked to Neon Postgres.")

            # 6. Compile Graph
            graph = create_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "legal_advisor_neon_sessions3"}}

            print("\n" + "="*50)
            print("⚖️  INDIAN LEGAL ADVISOR")
            print("="*50)
            print("Type 'exit' or 'quit' to end session.\n")

            while True:
                try:
                    user_input = input("User: ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ["exit", "quit", "bye"]:
                        break

                    logger.info(f"📥 Processing Query: '{user_input[:50]}...'")
                    input_state = {"messages": [HumanMessage(content=user_input)]}
                    
                    # Execute the graph stream
                    for event in graph.stream(input_state, config):
                        for node, value in event.items():
                            if node == "evaluator":
                                ai_msg = value["messages"][-1].content
                                score = value.get("evaluation_score", "N/A")
                                feedback = value.get("evaluation_feedback", "Verified via Scythe Path")
                                
                                # Use print for the user, logger for the audit trail
                                logger.info(f"\nAdvisor: {ai_msg}")
                                logger.info(f"📊 Quality Score: {score}/10 | {feedback}\n")

                except (EOFError, KeyboardInterrupt):
                    break
                except Exception as e:
                    # Specific handling for broken SSL pipes during a session
                    if "SSL connection" in str(e) or "closed" in str(e):
                        logger.error("🔌 Database connection lost. Re-establishing...")
                        # In a real app, you might want to re-initialize the pool here
                    else:
                        logger.error(f"⚠️ Error during conversation: {e}")
                    logger.info("\nAdvisor: I encountered an internal error. Please try again.\n")

    except Exception as e:
        logger.critical(f"🛑 CRITICAL BOOT FAILURE: {e}")
        sys.exit(1)
    finally:
        logger.info("🔒 Shutting down: Cleaning up resources.")
        logger.success("👋 Goodbye.")

if __name__ == "__main__":
    main()