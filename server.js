const express = require('express');
const { WebSocketServer } = require('ws');
const { Client } = require('@opensearch-project/opensearch');
const { PrismaClient } = require('@prisma/client');
const OpenAI = require('openai');
const dotenv = require('dotenv');
const path = require('path');
const fs = require('fs').promises;
const { v4: uuidv4 } = require('uuid');
const { VectorStoreRetrieverMemory } = require('langchain/memory');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const { OpenAIEmbeddings } = require('@langchain/openai');

// Load environment variables
dotenv.config();

const app = express();
const prisma = new PrismaClient();
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_URL,
});

// OpenSearch client setup
const osClient = new Client({
    node: `http://${process.env.OPENSEARCH_HOST || 'localhost'}:${process.env.OPENSEARCH_PORT || 9200}`,
    ssl: { rejectUnauthorized: false },
});

// LangChain memory setup
const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    configuration: { baseURL: process.env.OPENAI_EMBEDDINGS_URL },
    modelName: process.env.OPENAI_EMBED_MODEL || 'text-embedding-3-large',
});
const vectorStore = new MemoryVectorStore(embeddings);
const memory = new VectorStoreRetrieverMemory({
    vectorStore,
    memoryKey: 'chat_history',
    k: parseInt(process.env.TOP_K || 3),
});

// Configuration
const TOP_K = parseInt(process.env.TOP_K || 3);
const EMBED_DIM = parseInt(process.env.EMBED_DIM || 1024);
const UPLOAD_DIR = process.env.UPLOAD_DIR || '/usr/src/redmine/files';
const MAX_TERMS = parseInt(process.env.REDIS_MAX_ITEMS || 1000);
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
        // Mock user validation
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
                            project_id: { type: 'keyword' },
                            project_name: { type: 'text', fields: { keyword: { type: 'keyword' } } },
                            tracker_id: { type: 'keyword' },
                            status_id: { type: 'keyword' },
                            priority_id: { type: 'keyword' },
                            author_id: { type: 'keyword' },
                            author_name: { type: 'text', fields: { keyword: { type: 'keyword' } } },
                            subject: { type: 'text', analyzer: 'ticket_analyzer' },
                            description: { type: 'text', analyzer: 'ticket_analyzer' },
                            category_id: { type: 'keyword' },
                            category_name: { type: 'text', fields: { keyword: { type: 'keyword' } } },
                            created_on: { type: 'date', format: 'yyyy-MM-dd HH:mm:ss||strict_date_optional_time||epoch_millis' },
                            updated_on: { type: 'date', format: 'yyyy-MM-dd HH:mm:ss||strict_date_optional_time||epoch_millis' },
                            file_path: { type: 'keyword' },
                            file_type: { type: 'keyword' },
                            attachment_content: { type: 'text', analyzer: 'ticket_analyzer' },
                            semantic_text: { type: 'text' },
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
        model: process.env.OLLAMA_EMBED_MODEL || 'text-embedding-3-large',
        input: text,
    });
    return response.data[0].embedding;
}

// Retrieve Redmine document (issue, journal, or attachment)
async function retrieveRedmineDocument(docType, id, filePath) {
    if (docType === 'issue') {
        const issue = await prisma.issues.findUnique({
            where: { id: parseInt(id) },
            select: {
                id: true,
                project_id: true,
                tracker_id: true,
                status_id: true,
                priority_id: true,
                author_id: true,
                subject: true,
                description: true,
                created_on: true,
                updated_on: true,
                projects: { select: { name: true } },
                trackers: { select: { name: true } },
                issue_statuses: { select: { name: true } },
                enumerations: { select: { name: true } },
                users: { select: { login: true } },
            },
        });
        if (!issue) return null;
        return {
            doc_type: 'issue',
            content: {
                id: issue.id,
                project: issue.projects?.name,
                tracker: issue.trackers?.name,
                status: issue.issue_statuses?.name,
                priority: issue.enumerations?.name,
                author: issue.users?.login,
                subject: issue.subject,
                description: issue.description,
                created_on: issue.created_on,
                updated_on: issue.updated_on,
            },
        };
    } else if (docType === 'journal') {
        const journal = await prisma.journals.findUnique({
            where: { id: parseInt(id) },
            select: {
                id: true,
                journalized_id: true,
                notes: true,
                created_on: true,
                users: { select: { login: true } },
            },
        });
        if (!journal) return null;
        return {
            doc_type: 'journal',
            content: {
                id: journal.id,
                issue_id: journal.journalized_id,
                notes: journal.notes,
                created_on: journal.created_on,
                author: journal.users?.login,
            },
        };
    } else if (docType === 'attachment' && filePath) {
        const fullPath = path.resolve(UPLOAD_DIR, filePath);
        try {
            await fs.access(fullPath); // Verify file exists
            return { doc_type: 'attachment', file_path: fullPath };
        } catch (e) {
            console.error(`Attachment not found: ${fullPath}`, e);
            return null;
        }
    }
    return null;
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

// Planner: Generate query plan using GPT-4o
async function planQueries(query, chatHistory, indexName) {
    const tools = [
        {
            type: 'function',
            function: {
                name: 'generate_query_plan',
                description: 'Generates a dynamic plan for executing OpenSearch queries to retrieve all matching Redmine ticketing data.',
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
                                    purpose: { type: 'string', description: 'Purpose of the query (e.g., fetch issues, search attachments)' },
                                    requires_embedding: { type: 'boolean', description: 'Whether the query needs a semantic embedding' },
                                    depends_on: {
                                        type: 'array',
                                        items: { type: 'string' },
                                        description: 'Step IDs this query depends on (e.g., for issue IDs)',
                                    },
                                },
                                required: ['step_id', 'query_body', 'purpose', 'requires_embedding', 'depends_on'],
                            },
                            description: 'List of query steps to execute in order',
                        },
                    },
                    required: ['plan'],
                },
            },
        },
    ];

    const systemMsg = `
    You are an expert AI assistant for a Redmine ticketing system in medical informatics. Your task is to:
    1) Analyze the user query and generate a dynamic plan for executing OpenSearch queries to retrieve ALL matching documents.
    2) Produce a sequence of query steps, each with:
       - A unique step_id
       - A valid OpenSearch query_body (DSL) without size limits
       - A purpose describing the query's goal
       - A requires_embedding flag for semantic searches
       - A depends_on list for steps requiring data from previous queries (e.g., issue IDs)
    3) Ensure query bodies are minimal, valid OpenSearch DSL, and align with Redmine index mappings.
    4) Optimize queries by:
       - Using _source filtering for necessary fields only
       - Applying routing based on issue_id or project_id
       - Using terms lookup for lists exceeding ${MAX_TERMS}
    5) Support multi-step plans for complex queries (e.g., fetch issues, then attachments).
    6) Use ticket_analyzer for medical term synonyms (e.g., "billing issue" => "payment problem").
    7) Fetch ALL matching documents using scroll API, not limited by size.
    Index Mappings:
    - doc_id: keyword
    - doc_type: keyword
    - issue_id: keyword
    - project_id: keyword
    - project_name: text (keyword subfield)
    - tracker_id: keyword
    - status_id: keyword
    - priority_id: keyword
    - author_id: keyword
    - author_name: text (keyword subfield)
    - subject: text (ticket_analyzer)
    - description: text (ticket_analyzer)
    - category_id: keyword
    - category_name: text (keyword subfield)
    - created_on: date
    - updated_on: date
    - file_path: keyword
    - file_type: keyword
    - attachment_content: text (ticket_analyzer)
    - semantic_text: text
    - embedding: knn_vector (dimension: ${EMBED_DIM})
    Rules:
    - Output only the JSON object returned by the 'generate_query_plan' function.
    - Use chat history to inform the plan, but focus on the current query.
    - Avoid hallucinated fields or parameters.
    - Example:
      Query: "Find tickets about billing issues in project X created in 2023, and their PDF attachments"
      Output: {
        "plan": [
          {
            "step_id": "step1",
            "query_body": {
              "_source": ["issue_id", "project_id", "subject", "description"],
              "query": {
                "bool": {
                  "should": [
                    {"knn": {"embedding": {"vector": [], "k": 100}}},
                    {"match": {"semantic_text": "billing issues"}}
                  ],
                  "minimum_should_match": 1,
                  "filter": [
                    {"term": {"project_name.keyword": "X"}},
                    {"range": {"created_on": {"gte": "2023-01-01", "lte": "2023-12-31"}}}
                  ]
                }
              }
            },
            "purpose": "fetch all relevant issues",
            "requires_embedding": true,
            "depends_on": []
          },
          {
            "step_id": "step2",
            "query_body": {
              "_source": ["issue_id", "file_path", "file_type", "attachment_content"],
              "query": {
                "bool": {
                  "filter": [
                    {"terms": {"issue_id": []}},
                    {"term": {"file_type": "pdf"}}
                  ]
                }
              }
            },
            "purpose": "fetch all PDF attachments for matching issues",
            "requires_embedding": false,
            "depends_on": ["step1"]
          }
        ]
      }
  `;

    const prompt = `
    Query: ${query}
    Chat History: ${chatHistory}
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
    return result.plan;
}

// Executor: Execute query plan using scroll API
async function executeQueryPlan(plan, queryText, indexName, mappings, userId) {
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
                query_body.query.bool.filter = query_body.query.bool.filter.map(f => {
                    if (f.terms?.issue_id) {
                        if (issueIds.length > MAX_TERMS) {
                            const docId = await storeIdList(indexName, issueIds, 'issue_ids');
                            return { terms: { issue_id: { index: indexName, id: docId, path: 'issue_ids' } } };
                        }
                        return { terms: { issue_id: issueIds } };
                    }
                    return f;
                });
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
                routing: query_body.query?.bool?.filter?.find(f => f.term?.issue_id || f.term?.project_id)?.term?.issue_id ||
                    query_body.query?.bool?.filter?.find(f => f.term?.project_id)?.term?.project_id,
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

// Answerer: Synthesize results
async function generateAnswer(query, chatHistory, results, userId, chatId) {
    const retrievedDocs = [];
    for (const [step_id, result] of results.entries()) {
        if (result.aggregations) {
            return JSON.stringify(result.aggregations, null, 2);
        }
        for (const hit of result.hits) {
            const doc = hit._source;
            let docContent = null;
            if (doc.doc_type === 'issue') {
                docContent = await retrieveRedmineDocument('issue', doc.issue_id);
            } else if (doc.doc_type === 'journal') {
                docContent = await retrieveRedmineDocument('journal', doc.doc_id);
            } else if (doc.doc_type === 'attachment' && doc.file_path) {
                docContent = await retrieveRedmineDocument('attachment', null, doc.file_path);
                if (docContent) {
                    docContent.content = doc.attachment_content; // Use OpenSearch content
                }
            }
            if (docContent) {
                retrievedDocs.push({
                    doc_id: doc.doc_id,
                    doc_type: doc.doc_type,
                    issue_id: doc.issue_id,
                    project_id: doc.project_id,
                    content: docContent.content,
                    file_path: docContent.file_path,
                    file_type: doc.file_type,
                });
            }
        }
    }

    if (!retrievedDocs.length) {
        return 'No relevant documents found.';
    }

    // Build context
    const contextMap = {};
    retrievedDocs.forEach(doc => {
        const docId = doc.doc_id || 'UNKNOWN';
        let snippet = '';
        if (doc.doc_type === 'issue') {
            snippet = `[Issue #${doc.issue_id}] Subject: ${doc.content.subject} | Description: ${doc.content.description || ''}`;
        } else if (doc.doc_type === 'journal') {
            snippet = `[Journal #${doc.content.id} for Issue #${doc.issue_id}] Notes: ${doc.content.notes || ''}`;
        } else if (doc.doc_type === 'attachment') {
            snippet = `[Attachment for Issue #${doc.issue_id}] File: ${doc.file_path} | Content: ${doc.content?.slice(0, 200) || ''}...`;
        }
        contextMap[docId] = contextMap[docId] ? `${contextMap[docId]}\n${snippet}` : snippet;
    });

    const contextText = Object.entries(contextMap)
        .map(([docId, content]) => `--- Document ID: ${docId} ---\n${content}\n\n`)
        .join('');

    // Generate answer
    const systemMsg = `
    You are an AI assistant for a Redmine ticketing system in medical informatics. Rules:
    1) Cite document IDs as 'Document XYZ' and issue IDs as '#XYZ' without extensions.
    2) Every answer must reference document IDs or issue IDs used.
    3) If context is irrelevant, state 'I lack the context to answer your question.'
    4) Use only provided context and chat history, not external knowledge.
    5) If no context, admit it.
    6) Keep answers concise, max 4 sentences.
  `;

    const finalPrompt = `
    Chat History:\n${chatHistory}\n\n
    User Query:\n${query}\n\n
    Context:\n${contextText}\n
    --- End of context ---\n\n
    Provide your concise answer now.
  `;

    const response = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
            { role: 'system', content: systemMsg },
            { role: 'user', content: finalPrompt },
        ],
        max_tokens: 1024,
        temperature: 0.7,
    });

    // Include file paths in response for Redmine frontend
    const filePaths = retrievedDocs
        .filter(doc => doc.doc_type === 'attachment' && doc.file_path)
        .map(doc => ({ doc_id: doc.doc_id, issue_id: doc.issue_id, file_path: doc.file_path, file_type: doc.file_type }));

    return {
        answer: response.choices[0].message.content.trim(),
        file_paths: filePaths,
    };
}

// Main ask function (MCP loop)
async function ask(query, userId, chatId) {
    if (!query.trim()) throw new Error('Empty query');
    if (!chatId || !userId) throw new Error('Missing user_id or chat_id');

    // Authorization: Verify chat ownership
    const chat = await prisma.chat.findUnique({
        where: { id: chatId },
        include: { user: true },
    });
    if (!chat || chat.userId !== userId) {
        throw new Error('Chat not found or unauthorized');
    }

    const indexName = `${process.env.OPENSEARCH_INDEX_NAME || 'redmine'}-${userId}`;
    await ensureIndexExists(indexName);

    // RelevantHistoryRetriever: Fetch relevant chat history
    const chatHistoryDocs = await memory.getRelevantDocuments(query);
    const chatHistory = chatHistoryDocs.map(doc => doc.pageContent).join('\n');

    // Planner: Generate query plan
    const mappings = {
        properties: {
            doc_id: { type: 'keyword' },
            doc_type: { type: 'keyword' },
            issue_id: { type: 'keyword' },
            project_id: { type: 'keyword' },
            project_name: { type: 'text' },
            tracker_id: { type: 'keyword' },
            status_id: { type: 'keyword' },
            priority_id: { type: 'keyword' },
            author_id: { type: 'keyword' },
            author_name: { type: 'text' },
            subject: { type: 'text' },
            description: { type: 'text' },
            category_id: { type: 'keyword' },
            category_name: { type: 'text' },
            created_on: { type: 'date' },
            updated_on: { type: 'date' },
            file_path: { type: 'keyword' },
            file_type: { type: 'keyword' },
            attachment_content: { type: 'text' },
            semantic_text: { type: 'text' },
            embedding: { type: 'knn_vector', dimension: EMBED_DIM },
        },
    };

    const plan = await planQueries(query, chatHistory, indexName);

    // Executor: Execute query plan
    const results = await executeQueryPlan(plan, query, indexName, mappings, userId);

    // Answerer: Generate final answer
    const response = await generateAnswer(query, chatHistory, results, userId, chatId);

    // Store messages
    const currentTime = new Date().toISOString();
    await prisma.message.createMany({
        data: [
            { chatId, role: 'user', content: query, createdAt: currentTime },
            { chatId, role: 'assistant', content: response.answer, createdAt: currentTime },
        ],
    });

    // Store chat history in LangChain memory
    await memory.saveContext(
        { input: query },
        { output: response.answer }
    );

    return response;
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
        // WebSocket authentication
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

                // Authorization: Verify chat ownership
                const chat = await prisma.chat.findUnique({
                    where: { id: chat_id },
                    include: { user: true },
                });
                if (!chat || chat.userId !== user_id) {
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