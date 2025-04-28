/**********************************************************************
 *  index.js  – stateless-JWT + single index
 *********************************************************************/
const express = require('express');
const { WebSocketServer } = require('ws');
const { Client } = require('@opensearch-project/opensearch');
const OpenAI = require('openai');
const dotenv = require('dotenv');
const jwt = require('jsonwebtoken');
const cookie = require('cookie');

const { planAndExecute } = require('./agenticPlanner');
const runPlan = require('./executePlan');

dotenv.config();

/* ─────────────────── env & constants ─────────────────── */
const {
    OPENAI_API_KEY,
    OPENAI_API_URL,
    JWT_SECRET,                          //  HS256 secret or RSA/EdDSA public key
    OPENSEARCH_HOST = 'localhost',
    OPENSEARCH_PORT = 9200,
    OPENSEARCH_INDEX_NAME = 'redmine_index',
    BEARER_TOKEN_FALLBACK,               // keep old header-token for scripts optionally
    EMBED_DIM = 3072,
    SHARD_COUNT = 1,
    REPLICA_COUNT = 0,
} = process.env;

process.on('uncaughtException', (err) => {
    console.error('💥 Uncaught Exception:', err.stack || err);
    // graceful shutdown
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('💥 Unhandled Rejection at:', promise, 'reason:', reason);
    // graceful shutdown
    process.exit(1);
});


const app = express();

const openai = new OpenAI({
    apiKey: OPENAI_API_KEY,
    baseURL: OPENAI_API_URL,
});


const osClient = new Client({
    node: `http://${OPENSEARCH_HOST}:${OPENSEARCH_PORT}`,
    ssl: { rejectUnauthorized: false },
});

/* ─────────────────── middleware: JSON ─────────────────── */
app.use(express.json());

/* ─────────── JWT-only auth ────────── */
// function getPayloadFromRequest(req) {
//     // cookie (main auth)
//     if (req.headers.cookie) {
//         const cookies = cookie.parse(req.headers.cookie);
//         if (cookies.auth) {
//             return jwt.verify(cookies.auth, JWT_SECRET); // throws if invalid / expired
//         }
//     }
//     // fallback: Authorization: Bearer <legacy token>
//     const auth = req.headers.authorization || '';
//     if (auth.startsWith('Bearer ')) {
//         const token = auth.slice(7);
//         if (BEARER_TOKEN_FALLBACK && token === BEARER_TOKEN_FALLBACK) return { sub: 'legacy' };
//         return jwt.verify(token, JWT_SECRET);
//     }
//     throw new Error('JWT not found');
// }

// app.use((req, res, next) => {
//     try {
//         req.user = getPayloadFromRequest(req);   // { sub, email, role, … }
//         return next();
//     } catch (e) {
//         return res.status(401).json({ error: 'Authentication failed: ' + e.message });
//     }
// });

/* ───────────────── index bootstrap (single index) ────────────────── */
async function ensureIndexExists() {
    const exists = await osClient.indices.exists({ index: OPENSEARCH_INDEX_NAME });
    if (exists) return;

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
                    doc_type: { type: 'keyword' },
                    issue_id: { type: 'keyword' },
                    file_path: { type: 'keyword' },
                    file_type: { type: 'keyword' },
                    attachment_content: { type: 'text' },
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
}
ensureIndexExists().catch(console.error);

/* ───────────────── embedding helper ──────────────────── */
async function embedText(text) {
    if (!text.trim()) return Array(Number(EMBED_DIM)).fill(0);
    const { data } = await openai.embeddings.create({
        model: process.env.OPENAI_EMBED_MODEL || 'text-embedding-3-large',
        input: text,
    });
    return data[0].embedding;
}

// main query function - ask() 
async function ask(query) {
    if (!query?.trim()) throw new Error('Empty query');

    const mappings = {
        properties: {
            doc_id: { type: 'keyword' },
            doc_type: { type: 'keyword' },
            issue_id: { type: 'keyword' },
            file_path: { type: 'keyword' },
            file_type: { type: 'keyword' },
            attachment_content: { type: 'text' },
            embedding: { type: 'knn_vector', dimension: Number(EMBED_DIM) },
        },
    };

    const resMap = await planAndExecute({
        query,
        openai,
        osClient,
        indexName: OPENSEARCH_INDEX_NAME,
        mappings,
        embedText,
        runPlan,
    });

    const documents = [];
    for (const [, r] of resMap) for (const h of r.hits) documents.push(h._source);
    return { documents };
}

// Global error middleware for HTTP
app.use((err, req, res, next) => {
    console.error('Global Express Error Handler:', err.stack || err);
    res.status(err.status || 500).json({ error: err.message || 'Internal Server Error' });
});


/* ───────────────── REST: POST /ask ────────────────────── */
app.post('/ask', async (req, res) => {
    try {
        const { query } = req.body;
        if (!query) return res.status(400).json({ error: 'Missing query' });
        return res.json(await ask(query));
    } catch (e) {
        console.error(e);
        return res.status(500).json({ error: e.message });
    }
});

/* ───────────────── WebSocket: /ws/ask ────────────────── */
const wss = new WebSocketServer({ noServer: true });

wss.on('connection', (ws, req) => {
    // try {
    //     ws.user = getPayloadFromRequest(req);       // authenticate handshake
    // } catch (e) {
    //     ws.close(4401, 'unauthorized');             // 4401 = WS unauthorized
    //     return;
    // }

    ws.on('message', async data => {
        try {
            const { query } = JSON.parse(data);
            if (!query) throw new Error('Missing query');
            ws.send(JSON.stringify(await ask(query)));
        } catch (e) {
            console.error('WebSocket Message Error:', e.stack || e);
            ws.send(JSON.stringify({ error: e.message }));
        } finally {
            ws.close();
        }
    });
});

/* ─────────── HTTP→WS upgrade (only /ws/ask path) ─────── */
const server = app.listen(8000, () => console.log('API on http://localhost:8000'));
server.on('upgrade', (req, socket, head) => {
    if (req.url === '/ws/ask') {
        wss.handleUpgrade(req, socket, head, ws => wss.emit('connection', ws, req));
    } else socket.destroy();
});

/* ───────────────── graceful shutdown ─────────────────── */
process.on('SIGTERM', () => {
    console.log('Shutting down…');
    server.close(() => process.exit(0));
});
