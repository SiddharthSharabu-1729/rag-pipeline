import os
import subprocess
import random
from datetime import datetime, timedelta

# Configuration
DAYS_BACK = 90
COMMITS = [
    (
        "chore: initialize project and configuration",
        [".gitignore", ".python-version", "pyproject.toml", "uv.lock", ".env.example", "README.md", "src/rag/__init__.py"]
    ),
    (
        "feat: add base config and domain models",
        ["src/rag/config.py", "src/rag/domain/"]
    ),
    (
        "feat(ingestion): implement document parsers",
        ["src/rag/ingestion/parsers/"]
    ),
    (
        "feat(ingestion): add hierarchical chunking and ingestion pipeline",
        ["src/rag/ingestion/chunkers/", "src/rag/ingestion/pipeline.py"]
    ),
    (
        "feat(storage): setup vector database interfaces and embeddings",
        ["src/rag/storage/base.py", "src/rag/storage/embeddings.py"]
    ),
    (
        "feat(storage): implement qdrant store and sparse vectors",
        ["src/rag/storage/qdrant_store.py", "src/rag/storage/sparse.py"]
    ),
    (
        "feat(retrieval): build hybrid retriever and query transformations",
        ["src/rag/retrieval/hybrid_retriever.py", "src/rag/retrieval/query_transform.py"]
    ),
    (
        "feat(retrieval): add cross-encoder reranker and context optimizer",
        ["src/rag/retrieval/reranker.py", "src/rag/retrieval/context_optimizer.py"]
    ),
    (
        "feat(generation): configure llm client and prompt templates",
        ["src/rag/generation/llm_client.py", "src/rag/generation/prompt_templates.py"]
    ),
    (
        "feat(generation): add self-rag verifier and output guardrails",
        ["src/rag/generation/self_rag_verifier.py", "src/rag/generation/guardrails.py"]
    ),
    (
        "feat: wire up main RAG pipeline and semantic caching",
        ["src/rag/pipeline/", "src/rag/storage/semantic_cache.py"]
    ),
    (
        "feat(observability): integrate tracing telemetry",
        ["src/rag/observability/"]
    ),
    (
        "feat(api): build FastAPI routes and health checks",
        ["src/rag/api/"]
    ),
    (
        "feat(eval): add evaluation metrics and scoring logic",
        ["src/rag/evaluation/"]
    ),
    (
        "feat: add CLI interface",
        ["src/rag/cli.py"]
    ),
    (
        "test: add unit tests for storage and ingestion",
        ["tests/test_storage.py", "tests/test_ingestion.py"]
    ),
    (
        "test: add tests for retrieval, generation, and api",
        ["tests/test_retrieval_and_reranking.py", "tests/test_guardrails_and_self_rag.py", "tests/test_observability_and_api.py"]
    ),
    (
        "test: add end-to-end testing suite",
        ["tests/test_end_to_end.py"]
    ),
    (
        "chore: add dockerization and finalize documentation",
        ["Dockerfile", "docker-compose.yml", "article.md"]
    ),
    (
        "fix: minor bug fixes and final cleanups",
        ["."] # Catch-all for any remaining uncommitted files
    )
]

def run_cmd(cmd, env=None):
    # Removed DEVNULL so errors print directly to your terminal
    subprocess.run(cmd, shell=True, env=env)

def generate_history():
    if os.path.exists(".git"):
        print("Cleaning up old .git folder...")
        # Force remove the .git directory on Windows
        subprocess.run('rmdir /S /Q .git', shell=True)

    print("Initializing git repository...")
    run_cmd("git init")

    start_date = datetime.now() - timedelta(days=DAYS_BACK)
    current_date = start_date

    print(f"Recreating history with {len(COMMITS)} commits...\n")
    
    for i, (message, paths) in enumerate(COMMITS):
        days_to_add = random.uniform(2.0, 5.0)
        current_date += timedelta(days=days_to_add)
        
        current_date = current_date.replace(
            hour=random.randint(9, 23),
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )
        
        # Git standard date format
        date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")

        added_something = False
        for path in paths:
            # Safely handle Windows paths (changes / to \)
            safe_path = os.path.normpath(path)
            
            # Only attempt to add if the file/folder actually exists
            if os.path.exists(safe_path) or safe_path == ".":
                run_cmd(f'git add "{safe_path}"')
                added_something = True
            else:
                print(f"  [Warning] Missing path skipped: {safe_path}")

        if not added_something:
            print(f"  [Skipped] Commit '{message}' (No files found to stage)")
            continue

        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = date_str
        env["GIT_COMMITTER_DATE"] = date_str

        run_cmd(f'git commit -m "{message}"', env=env)
        print(f"Committed: '{message}' at {date_str}\n")

    print("Done! Check the Source Control tab in VS Code or run 'git log'.")

if __name__ == "__main__":
    generate_history()