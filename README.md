# 🧠 RASS Engine Backend

A dynamic, LLM-based Retrieval-Augmented Generation (RAG) or Retrieval-Augmented Semantic Search (RASS) engine backend for intelligent search over files/documents. This system interprets natural language queries, generates multi-hop embedding search plans, and retrieves semantically relevant documents via OpenSearch and OpenAI embeddings.

---

``` mermaid
graph TD
    A[User Query via REST or WebSocket] --> B[Planner using GPT-4o]
    B --> C[Intent and Entity Extraction]
    C --> D[ANN Plan Generation]
    D --> E[Plan and Execute Loop]
    E --> F[Embed Search Terms via OpenAI]
    F --> G[Run ANN Search with OpenSearch]
    G --> H[Filter Hits by Score Threshold]
    H --> I[Group Hits Round-Robin]
    I --> J[Check Coverage via LLM]
    J --> K{Is Coverage Sufficient?}
    K -->|Yes| L[Return Top-k Grouped Results]
    K -->|No| E
    L --> M[Send JSON or WebSocket Response]

```

## ⚙️ Core Features

* **Agentic/LLM-based Planner + Executor Loop**
  Leverages GPT-4o to extract entities, expand terms, and build a multi-step ANN plan from natural queries.

* **ANN Search via OpenSearch (HNSW)**
  Efficient vector search over document embeddings using 'knn_vector' with score threshold filtering.

* **Interleaved Multi-Entity Result Retrieval**
  Returns top-k interleaved results across query entities (X1, Y1, Z1, X2 ...) to ensure balanced relevance.

* **WebSocket + REST API Support**
  Real-time and stateless access methods for seamless front-end integration.

---

## 🧰 Tech Stack

| Component       | Technology                        |
| --------------- | --------------------------------- |
| API Server      | Node.js + Express.js              |
| Embeddings      | OpenAI (text-embedding-ada-002)   |
| Search Engine   | OpenSearch (HNSW, KNN)            |
| Planner Model   | GPT-4o                            |
| Socket Support  | WebSocket Server (WSS)            |
| Auth            | JWT                               |

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
git clone https://github.com/NeuralRevenant/rass-engine.git
cd rass-engine
npm install
```

### 2. Configure Environment Variables

Create a `.env` file:

```ini
# OpenAI Config
OPENAI_API_KEY=sk-...
OPENAI_API_URL=https://api.openai.com/v1
OPENAI_EMBED_MODEL=text-embedding-ada-002

# OpenSearch Config
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX_NAME=redmine_index
EMBED_DIM=1536
DEFAULT_K=25
```

### 3. Start the Server

```bash
node server.js
```

Server runs on: [http://localhost:8000](http://localhost:8000)

---

## 🧠 How It Works

* **Natural Language Input** → `"get me records for Julian140 and documents with Borne"`
* **LLM Planner** → Extracts entities ('Julian140', 'Borneo Elephants'), expands them (if needed), and plans vector queries
* **Embedding Search** → Each entity/term is searched via HNSW-based ANN
* **Results** → Interleaved top results returned, preserving per-entity relevance

---

## 🔌 REST API

### `POST /ask`

**Request**

```json
{
  "query": "get me records having the term Julian140 and the documents containing the term Borne",
  "top_k": 5
}
```

**Response**

```json
{
  "documents": [
    {
      "doc_id": "09a34661-e7c6-44b6-b068-303dd8df8b1b_000bdad9-dc9a-49ba-b1ac-980d4e18ca08.json-0",
      "file_path": "/.../uploads/...ca08.json",
      "file_type": "json",
      "score": 0.8682902
    },
    {
      "doc_id": "aebcce94-8bff-4058-85c6-9371e92f35ad_PMC176546.txt-1",
      "file_path": "/.../uploads/...546.txt",
      "file_type": "txt",
      "score": 0.88842297
    },
    {
      "doc_id": "2abd50c3-4483-4ad4-a9f9-70d509c506e8_000d4013-e5e1-441b-bdc4-5fca55dbe565.json-0",
      "file_path": "/.../uploads/...6565.json",
      "file_type": "json",
      "score": 0.8680167
    },
    {
      "doc_id": "aebcce94-8bff-4058-85c6-9371e92f35ad_PMC176546.txt-0",
      "file_path": "/.../uploads/...546.txt",
      "file_type": "txt",
      "score": 0.88662094
    },
    {
      "doc_id": "0bd0a0ad-b520-420a-b66a-c5dfc0217468_000e1a87-e036-42bc-9cbe-e5ffcf61acb4.json-1",
      "file_path": "/.../uploads/...acb4.json",
      "file_type": "json",
      "score": 0.86678123
    }
  ]
}
```

---

## 🌐 WebSocket API

### `ws://localhost:8000/ws/ask`

**Request Message**

```json
{
  "query": "heart disease and asthma in California",
  "top_k": 5
}
```

**Response Message**

Same structure as the REST `/ask` response. Connection auto-closes post-response.

---

## 🧼 Notes

* Score threshold is configurable (default ≥ `0.86`) and applied per KNN step.
* ANN results are interleaved by entity, not globally sorted, to preserve diverse entity coverage.
* Embeddings are cached per query step to avoid recomputation.
* Planner auto-retries up to 6 times if no adequate coverage is achieved.

---

## 📌 Future Enhancements for medical EHR document search

* Improve the accuracy for EHR patient and medical data - FHIR, plain-text medical notes with hybrid search and proper file parsing.
* Add a more powerful agentic AI design to enhance the retrieval accuracy like above for EHR medical documents.
* Enable hybrid KNN + text-based search (BM25, etc. used in OpenSearch)
* Visual result explorer (timeline or graph)
