const express = require('express');
const { WebSocketServer } = require('ws');
const { Client } = require('@opensearch-project/opensearch');
const { PrismaClient } = require('@prisma/client');
const OpenAI = require('openai');
const dotenv = require('dotenv');
const path = require('path');
const fs = require('fs').promises;
const { v4: uuidv4 } = require('uuid');

// Load environment variables
dotenv.config();

const app = express();
const prisma = new PrismaClient();
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_COMPLETIONS_URL,
});

// OpenSearch client setup
const osClient = new Client({
    node: `http://${process.env.OPENSEARCH_HOST || 'localhost'}:${process.env.OPENSEARCH_PORT || 9200}`,
    ssl: { rejectUnauthorized: false },
});

// Configuration
const EMBED_DIM = parseInt(process.env.EMBED_DIM || 3072);
const UPLOAD_DIR = process.env.UPLOAD_DIR || '/uploads';
const MAX_TERMS = 1000;
const SHARD_COUNT = parseInt(process.env.SHARD_COUNT || 1);
const REPLICA_COUNT = parseInt(process.env.REPLICA_COUNT || 0);
const SCROLL_TIMEOUT = '1m';

// Middleware
app.use(express.json());

// Authentication middleware
const authenticateUser = async (req, res, next) => {
    const authHeader = req.headers['authorization'];
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({ error: 'Missing or invalid Authorization header' });
    }
    const token = authHeader.split(' ')[1];
    try {
        if (token !== process.env.BEARER_TOKEN) {
            throw new Error('Invalid token');
        }
        req.user = { id: req.body.user_id || req.query.user_id };
        if (!req.user.id) {
            throw new Error('Missing user_id');
        }
        next();
    } catch (e) {
        res.status(401).json({ error: 'Authentication failed: ' + e.message });
    }
};
app.use(authenticateUser);

// OpenSearch index setup
async function ensureIndexExists(indexName) {
    try {
        const exists = await osClient.indices.exists({ index: indexName });
        if (!exists) {
            await osClient.indices.create({
                index: indexName,
                body: {
                    settings: {
                        index: {
                            knn: true,
                            number_of_shards: SHARD_COUNT,
                            number_of_replicas: REPLICA_COUNT,
                        },
                        analysis: {
                            filter: {
                                ticket_synonym: {
                                    type: 'synonym',
                                    synonyms: [
                                        'billing issue, payment problem, invoice error',
                                        'patient data, medical record, ehr data',
                                        'privacy issue, data breach, confidentiality',
                                    ],
                                },
                            },
                            analyzer: {
                                ticket_analyzer: {
                                    tokenizer: 'standard',
                                    filter: ['lowercase', 'ticket_synonym'],
                                },
                            },
                        },
                    },
                    mappings: {
                        properties: {
                            doc_id: { type: 'keyword' },
                            doc_type: { type: 'keyword' },
                            issue_id: { type: 'keyword' },
                            file_path: { type: 'keyword' },
                            file_type: { type: 'keyword' },
                            attachment_content: { type: 'text', analyzer: 'ticket_analyzer' },
                            embedding: {
                                type: 'knn_vector',
                                dimension: EMBED_DIM,
                                method: { name: 'hnsw', engine: 'nmslib', space_type: 'cosinesimil', parameters: { m: 48, ef_construction: 400 } },
                            },
                        },
                    },
                },
            });
            console.log(`Created index '${indexName}' in OpenSearch.`);
        }
    } catch (e) {
        console.error(`Failed to create index '${indexName}':`, e);
        throw e;
    }
}

// Store large ID lists in OpenSearch for terms lookup
async function storeIdList(indexName, ids, field) {
    const docId = `${field}_list_${uuidv4()}`;
    await osClient.index({
        index: indexName,
        id: docId,
        body: { [field]: ids },
    });
    return docId;
}

// Embedding function using OpenAI
async function embedText(text) {
    if (!text.trim()) return new Array(EMBED_DIM).fill(0);
    const response = await openai.embeddings.create({
        model: process.env.OPENAI_EMBED_MODEL || 'text-embedding-3-large',
        input: text,
    });
    return response.data[0].embedding;
}

// Validate OpenSearch query against mappings
function validateOpensearchQuery(queryBody, mappings) {
    const validFields = Object.keys(mappings.properties);
    function traverseQuery(obj) {
        for (const key in obj) {
            if (key === 'knn' && obj[key].embedding) {
                if (!validFields.includes('embedding') || obj[key].embedding.vector?.length !== EMBED_DIM) {
                    throw new Error('Invalid knn vector or field');
                }
            } else if (['match', 'term', 'terms', 'range'].includes(key)) {
                const field = Object.keys(obj[key])[0];
                if (!validFields.includes(field.split('.')[0])) {
                    throw new Error(`Invalid field: ${field}`);
                }
            } else if (typeof obj[key] === 'object') {
                traverseQuery(obj[key]);
            }
        }
    }
    traverseQuery(queryBody);
    return true;
}

// Planner: Generate query plan using GPT-4o with looping
async function planQueries(query, indexName, previousResults = []) {
    const tools = [
        {
            type: 'function',
            function: {
                name: 'generate_query_plan',
                description: 'Generates a dynamic plan for executing OpenSearch queries to retrieve matching documents with file_path.',
                parameters: {
                    type: 'object',
                    properties: {
                        plan: {
                            type: 'array',
                            items: {
                                type: 'object',
                                properties: {
                                    step_id: { type: 'string', description: 'Unique ID for the step' },
                                    query_body: {
                                        type: 'object',
                                        description: 'OpenSearch query body in JSON format',
                                        additionalProperties: true,
                                    },
                                    purpose: { type: 'string', description: 'Purpose of the query' },
                                    requires_embedding: { type: 'boolean', description: 'Whether the query needs a semantic embedding' },
                                    depends_on: {
                                        type: 'array',
                                        items: { type: 'string' },
                                        description: 'Step IDs this query depends on',
                                    },
                                    is_final: { type: 'boolean', description: 'Whether this is the final step' },
                                },
                                required: ['step_id', 'query_body', 'purpose', 'requires_embedding', 'depends_on', 'is_final'],
                            },
                        },
                    },
                    required: ['plan'],
                },
            },
        },
    ];

    const systemMsg = `
    You are an AI assistant for a document retrieval system. Your task is to:
    1) Analyze the user query and generate a dynamic plan for OpenSearch queries to retrieve documents with a file_path.
    2) Produce a sequence of query steps, each with:
       - step_id: Unique ID
       - query_body: Valid OpenSearch DSL (use knn for embeddings)
       - purpose: Query goal
       - requires_embedding: True if knn is used
       - depends_on: Step IDs for dependencies (e.g., issue IDs)
       - is_final: True if no further steps are needed
    3) Support chained queries where one step’s output (e.g., issue IDs) feeds into the next.
    4) Use terms lookup for lists exceeding ${MAX_TERMS}.
    5) Fetch ALL matching documents using scroll API.
    Index Mappings:
    - doc_id: keyword
    - doc_type: keyword
    - issue_id: keyword
    - file_path: keyword
    - file_type: keyword
    - attachment_content: text (ticket_analyzer)
    - embedding: knn_vector (dimension: ${EMBED_DIM})
    Rules:
    - Output only the JSON from 'generate_query_plan'.
    - Use previous results to inform chaining.
    - Set is_final: true when the plan retrieves all required documents.
    Example:
      Query: "Find PDF attachments for billing issues"
      Previous Results: [{ step_id: "step1", hits: [{ _source: { issue_id: "123" } }] }]
      Output: {
        "plan": [
          {
            "step_id": "step2",
            "query_body": {
              "_source": ["doc_id", "file_path", "file_type"],
              "query": {
                "bool": {
                  "filter": [
                    {"terms": {"issue_id": ["123"]}},
                    {"term": {"file_type": "pdf"}}
                  ]
                }
              }
            },
            "purpose": "fetch PDF attachments",
            "requires_embedding": false,
            "depends_on": ["step1"],
            "is_final": true
          }
        ]
      }
  `;

    let allPlans = [];
    let currentResults = previousResults;

    while (true) {
        const previousContext = currentResults.length
            ? JSON.stringify(currentResults.map(r => ({ step_id: r.step_id, hits: r.hits.map(h => h._source) })))
            : '[]';

        const prompt = `
      Query: ${query}
      Previous Results: ${previousContext}
      Generate the query plan using the 'generate_query_plan' function.
    `;

        const response = await openai.chat.completions.create({
            model: 'gpt-4o',
            messages: [
                { role: 'system', content: systemMsg },
                { role: 'user', content: prompt },
            ],
            tools: tools,
            tool_choice: { type: 'function', function: { name: 'generate_query_plan' } },
        });

        const result = JSON.parse(response.choices[0].message.tool_calls[0].function.arguments);
        const newPlan = result.plan;

        allPlans.push(...newPlan);

        if (newPlan.some(step => step.is_final)) {
            break;
        }

        // Execute the new plan to get results for the next iteration
        const newResults = await executeQueryPlan(newPlan, query, indexName, {
            properties: {
                doc_id: { type: 'keyword' },
                doc_type: { type: 'keyword' },
                issue_id: { type: 'keyword' },
                file_path: { type: 'keyword' },
                file_type: { type: 'keyword' },
                attachment_content: { type: 'text' },
                embedding: { type: 'knn_vector', dimension: EMBED_DIM },
            },
        });
        currentResults = Array.from(newResults.entries()).map(([step_id, result]) => ({ step_id, ...result }));
    }

    return allPlans;
}

// Executor: Execute query plan using scroll API
async function executeQueryPlan(plan, queryText, indexName, mappings) {
    const results = new Map();
    let queryEmb = null;

    for (const step of plan) {
        const { step_id, query_body, purpose, requires_embedding, depends_on } = step;

        // Check dependencies
        for (const depId of depends_on) {
            if (!results.has(depId)) {
                console.warn(`Skipping step ${step_id}: Missing dependency ${depId}`);
                continue;
            }
        }

        // Embed query if required
        if (requires_embedding && !queryEmb) {
            queryEmb = await embedText(queryText);
        }

        // Inject embedding into knn clauses
        if (requires_embedding && queryEmb) {
            function injectEmbedding(obj) {
                if (obj.knn?.embedding) {
                    obj.knn.embedding.vector = queryEmb;
                }
                for (const key in obj) {
                    if (typeof obj[key] === 'object') {
                        injectEmbedding(obj[key]);
                    }
                }
            }
            injectEmbedding(query_body);
        }

        // Inject issue IDs from dependencies
        if (query_body.query?.bool?.filter?.some(f => f.terms?.issue_id)) {
            const issueIds = Array.from(results.entries())
                .filter(([id]) => depends_on.includes(id))
                .flatMap(([_, res]) => res.hits.map(hit => hit._source.issue_id))
                .filter(id => id);
            if (issueIds.length) {
                let termsFilter;
                if (issueIds.length > MAX_TERMS) {
                    const docId = await storeIdList(indexName, issueIds, 'issue_ids');
                    termsFilter = { terms: { issue_id: { index: indexName, id: docId, path: 'issue_ids' } } };
                } else {
                    termsFilter = { terms: { issue_id: issueIds } };
                }
                query_body.query.bool.filter = query_body.query.bool.filter.map(f =>
                    f.terms?.issue_id ? termsFilter : f
                );
            }
        }

        // Validate query
        try {
            validateOpensearchQuery(query_body, mappings);
        } catch (e) {
            console.error(`Invalid query for step ${step_id} (${purpose}):`, e.message);
            continue;
        }

        // Execute query with scroll API
        try {
            const allHits = [];
            let scrollId;

            // Initial search with scroll
            const initialResult = await osClient.search({
                index: indexName,
                body: query_body,
                scroll: SCROLL_TIMEOUT,
            });

            allHits.push(...initialResult.hits.hits);
            scrollId = initialResult._scroll_id;

            // Continue scrolling until no more results
            while (true) {
                const scrollResult = await osClient.scroll({
                    scroll_id: scrollId,
                    scroll: SCROLL_TIMEOUT,
                });
                if (!scrollResult.hits.hits.length) break;
                allHits.push(...scrollResult.hits.hits);
            }

            // Clear scroll
            if (scrollId) {
                await osClient.clearScroll({ scroll_id: [scrollId] });
            }

            results.set(step_id, { purpose, hits: allHits, aggregations: initialResult.aggregations });
        } catch (e) {
            console.error(`Error executing step ${step_id} (${purpose}):`, e);
        }
    }

    return results;
}

// Optional: Retrieve documents from disk
async function retrieveFromDisk(filePath) {
    const fullPath = path.resolve(UPLOAD_DIR, filePath);
    try {
        await fs.access(fullPath);
        const content = await fs.readFile(fullPath, 'utf8');
        return { file_path: fullPath, content };
    } catch (e) {
        console.error(`File not found: ${fullPath}`, e);
        return null;
    }
}

// Main ask function
async function ask(query, userId, chatId) {
    if (!query.trim()) throw new Error('Empty query');
    if (!chatId || !userId) throw new Error('Missing user_id or chat_id');

    // Authorization: Verify chat ownership
    const chat = await prisma.chat.findUnique({
        where: { id: chatId },
        include: { user: true },
    });
    if (!chat || chat.userId !== parseInt(userId)) {
        throw new Error('Chat not found or unauthorized');
    }

    const indexName = `${process.env.OPENSEARCH_INDEX_NAME || 'redmine'}-${userId}`;
    await ensureIndexExists(indexName);

    // Generate query plan with looping
    const plan = await planQueries(query, indexName);

    // Execute query plan
    const results = await executeQueryPlan(plan, query, indexName, {
        properties: {
            doc_id: { type: 'keyword' },
            doc_type: { type: 'keyword' },
            issue_id: { type: 'keyword' },
            file_path: { type: 'keyword' },
            file_type: { type: 'keyword' },
            attachment_content: { type: 'text' },
            embedding: { type: 'knn_vector', dimension: EMBED_DIM },
        },
    });

    // Collect results with file_path
    const retrievedDocs = [];
    for (const [step_id, result] of results.entries()) {
        if (result.aggregations) {
            retrievedDocs.push({ step_id, aggregations: result.aggregations });
        } else {
            for (const hit of result.hits) {
                const doc = hit._source;
                if (doc.file_path) {
                    retrievedDocs.push({
                        step_id,
                        doc_id: doc.doc_id,
                        doc_type: doc.doc_type,
                        issue_id: doc.issue_id,
                        file_path: doc.file_path,
                        file_type: doc.file_type,
                        attachment_content: doc.attachment_content,
                    });
                }
            }
        }
    }

    return { documents: retrievedDocs };
}

// HTTP endpoint
app.post('/ask', async (req, res) => {
    try {
        const { query, user_id, chat_id } = req.body;
        if (!query || !user_id || !chat_id) {
            return res.status(400).json({ error: 'Provide user_id, chat_id, query' });
        }

        const response = await ask(query, user_id, chat_id);
        res.json(response);
    } catch (e) {
        console.error('Error in /ask:', e);
        res.status(500).json({ error: e.message });
    }
});

// WebSocket endpoint
const wss = new WebSocketServer({ noServer: true });
wss.on('connection', async (ws, request) => {
    try {
        const authHeader = request.headers['authorization'];
        if (!authHeader || !authHeader.startsWith('Bearer ')) {
            ws.send(JSON.stringify({ error: 'Missing or invalid Authorization header' }));
            ws.close();
            return;
        }
        const token = authHeader.split(' ')[1];
        if (token !== process.env.BEARER_TOKEN) {
            ws.send(JSON.stringify({ error: 'Authentication failed' }));
            ws.close();
            return;
        }

        ws.on('message', async (data) => {
            try {
                const { query, user_id, chat_id } = JSON.parse(data);
                if (!query || !user_id || !chat_id) {
                    ws.send(JSON.stringify({ error: 'Missing required parameters.' }));
                    ws.close();
                    return;
                }

                const chat = await prisma.chat.findUnique({
                    where: { id: chat_id },
                    include: { user: true },
                });
                if (!chat || chat.userId !== parseInt(user_id)) {
                    ws.send(JSON.stringify({ error: 'Chat not found or unauthorized access.' }));
                    ws.close();
                    return;
                }

                const response = await ask(query, user_id, chat_id);
                ws.send(JSON.stringify(response));
                ws.close();
            } catch (e) {
                console.error('WebSocket error:', e);
                ws.send(JSON.stringify({ error: 'Internal server error occurred.' }));
                ws.close();
            }
        });
    } catch (e) {
        console.error('WebSocket connection error:', e);
        ws.send(JSON.stringify({ error: 'Internal server error occurred.' }));
        ws.close();
    }
});

// HTTP server upgrade for WebSocket
const server = app.listen(8000, () => {
    console.log('Server running on http://localhost:8000');
});

server.on('upgrade', (request, socket, head) => {
    if (request.url === '/ws/ask') {
        wss.handleUpgrade(request, socket, head, (ws) => {
            wss.emit('connection', ws, request);
        });
    } else {
        socket.destroy();
    }
});

// Graceful shutdown
process.on('SIGTERM', async () => {
    console.log('Shutting down...');
    await prisma.$disconnect();
    server.close(() => {
        console.log('Server closed.');
        process.exit(0);
    });
});