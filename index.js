const express = require('express');
const { WebSocketServer } = require('ws');
const { Client } = require('@opensearch-project/opensearch');
const OpenAI = require('openai');
const dotenv = require('dotenv');
const cookie = require('cookie');
const jwt = require('jsonwebtoken');

const { planAndExecute } = require('./agenticPlanner');
const { runSteps } = require('./executePlan');

dotenv.config();

const {
    OPENAI_API_KEY,
    OPENAI_API_URL = 'https://api.openai.com/v1',
    OPENSEARCH_HOST = 'localhost',
    OPENSEARCH_PORT = 9200,
    OPENSEARCH_INDEX_NAME = 'redmine_index',
    OPENAI_EMBED_MODEL = 'text-embedding-ada-002',
    EMBED_DIM = 1536,
    SHARD_COUNT = 1,
    REPLICA_COUNT = 0,
    DEFAULT_TOP_K = 25,
} = process.env;

const app = express();
const openai = new OpenAI({ apiKey: OPENAI_API_KEY, baseURL: OPENAI_API_URL });
const osClient = new Client({
    node: `http://${OPENSEARCH_HOST}:${OPENSEARCH_PORT}`,
    ssl: { rejectUnauthorized: false }
});

app.use(express.json());

/**
 * Ensures the OpenSearch index exists with updated mappings.
 */
async function ensureIndexExists() {
    try {
        const exists = await osClient.indices.exists({ index: OPENSEARCH_INDEX_NAME });
        if (exists.body) {
            console.log(`Index ${OPENSEARCH_INDEX_NAME} already exists`);
            return;
        }

        await osClient.indices.create({
            index: OPENSEARCH_INDEX_NAME,
            body: {
                settings: {
                    index: {
                        knn: true,
                        number_of_shards: SHARD_COUNT,
                        number_of_replicas: REPLICA_COUNT,
                    },
                },
                mappings: {
                    properties: {
                        doc_id: { type: 'keyword' },
                        file_path: { type: 'keyword' },
                        file_type: { type: 'keyword' },
                        text_chunk: { type: 'text' },
                        embedding: {
                            type: 'knn_vector',
                            dimension: Number(EMBED_DIM),
                            method: {
                                name: 'hnsw',
                                engine: 'nmslib',
                                space_type: 'cosinesimil',
                                parameters: { m: 48, ef_construction: 400 },
                            },
                        },
                    },
                },
            },
        });
        console.log(`Created index ${OPENSEARCH_INDEX_NAME}`);
    } catch (err) {
        console.error('Failed to create index:', err.message);
        throw err;
    }
}

// Generates embeddings for entities
async function embedText(text) {
    if (!text?.trim()) throw new Error('Empty text for embedding');
    try {
        const { data } = await openai.embeddings.create({
            model: OPENAI_EMBED_MODEL,
            input: text.slice(0, 14000),
        });
        if (data[0].embedding.length !== Number(EMBED_DIM)) {
            throw new Error(`Invalid embedding dimension: ${data[0].embedding.length}`);
        }
        console.log(`Generated embedding for "${text}": length=${data[0].embedding.length}`);
        return data[0].embedding;
    } catch (err) {
        console.error(`Embedding error for "${text}":`, err.message);
        throw err;
    }
}

// Main query function
async function ask(query, top_k) {
    if (!query?.trim()) throw new Error('Empty query');

    // Check index health and document count
    try {
        const health = await osClient.cluster.health();
        console.log('OpenSearch cluster health:', health.body.status);
        const count = await osClient.count({ index: OPENSEARCH_INDEX_NAME });
        console.log(`Index ${OPENSEARCH_INDEX_NAME} - total document count: ${count.body.count}`);
    } catch (err) {
        console.error('OpenSearch health check failed:', err.message);
    }

    const mappings = {
        properties: {
            doc_id: { type: 'keyword' },
            file_path: { type: 'keyword' },
            file_type: { type: 'keyword' },
            text_chunk: { type: 'text' },
            embedding: { type: 'knn_vector', dimension: Number(EMBED_DIM) },
        },
    };

    const hits = await planAndExecute({
        query,
        openai,
        osClient,
        indexName: OPENSEARCH_INDEX_NAME,
        mappings,
        embedText,
        runStepsFn: runSteps,
    });

    const documents = hits.map(h => ({
        doc_id: h._source.doc_id,
        file_path: h._source.file_path,
        file_type: h._source.file_type,
        // text: h._source.text_chunk,
        score: h._score || 1.0
    }));

    if (!documents.length) {
        throw new Error('No matching documents found for the query.');
    }

    return { documents: documents.slice(0, top_k || DEFAULT_TOP_K) };
}

// API endpoints
app.post('/ask', async (req, res) => {
    try {
        const { query, top_k } = req.body;
        if (!query) return res.status(400).json({ error: 'Missing query' });
        console.log('--------------------------------- Processing query: ', query, '-------------------------------');
        return res.json(await ask(query, top_k));
    } catch (e) {
        console.error('Ask endpoint error:', e);
        return res.status(500).json({ error: e.message });
    }
});

const wss = new WebSocketServer({ noServer: true });
wss.on('connection', (ws) => {
    ws.on('message', async msg => {
        try {
            const { query, top_k } = JSON.parse(msg);
            if (!query) throw new Error('Missing query');
            ws.send(JSON.stringify(await ask(query, top_k)));
        } catch (e) {
            console.error('WebSocket error:', e);
            ws.send(JSON.stringify({ error: e.message }));
        } finally {
            ws.close();
        }
    });
});

async function startServer() {
    try {
        await ensureIndexExists();
        const srv = app.listen(8000, () => console.log('API on http://localhost:8000'));
        srv.on('upgrade', (req, sock, head) => {
            if (req.url === '/ws/ask') {
                wss.handleUpgrade(req, sock, head, ws => wss.emit('connection', ws, req));
            } else {
                sock.destroy();
            }
        });
    } catch (e) {
        console.error('Failed to start server:', e);
        process.exit(1);
    }
}

startServer();

process.on('SIGTERM', () => {
    console.log('Shutting down…');
    process.exit(0);
});