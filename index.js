const express = require('express');
const { WebSocketServer } = require('ws');
const { Client } = require('@opensearch-project/opensearch');
const { PrismaClient } = require('@prisma/client');
const OpenAI = require('openai');
const Redis = require('redis');
const dotenv = require('dotenv');
const path = require('path');
const fs = require('fs').promises;
const { v4: uuidv4 } = require('uuid');
const pdfParse = require('pdf-parse');
const docx2txt = require('docx2txt');

// Load environment variables
dotenv.config();

const app = express();
const prisma = new PrismaClient();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const redisClient = Redis.createClient({ url: process.env.REDIS_URL || 'redis://localhost:6379' });

// OpenSearch client setup
const osClient = new Client({
    node: `http://${process.env.OPENSEARCH_HOST || 'localhost'}:${process.env.OPENSEARCH_PORT || 9200}`,
    ssl: { rejectUnauthorized: false },
});

// Configuration
const TOP_K = parseInt(process.env.TOP_K || 5);
const MAX_CHAT_HISTORY = parseInt(process.env.MAX_CHAT_HISTORY || 10);
const EMBED_DIM = 3072; // text-embedding-3-large dimension
const MAX_FILES_PER_ISSUE = parseInt(process.env.MAX_FILES_PER_ISSUE || 5);
const UPLOAD_DIR = process.env.UPLOAD_DIR || '/path/to/redmine/files';
const SUPPORTED_FILE_EXTENSIONS = ['.pdf', '.txt', '.md', '.json', '.docx'];
const CACHE_TTL = parseInt(process.env.CACHE_TTL || 3600); // 1 hour
const MAX_TERMS = parseInt(process.env.MAX_TERMS || 1000);

// Middleware
app.use(express.json());

// Connect Redis
redisClient.connect().catch(err => console.error('Redis connection error:', err));

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
                            number_of_shards: parseInt(process.env.SHARD_COUNT || 1),
                            number_of_replicas: parseInt(process.env.REPLICA_COUNT || 0),
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
                            doc_type: { type: 'keyword', enum: ['issue', 'journal', 'attachment'] },
                            issue_id: { type: 'keyword' },
                            project_id: { type: 'keyword' },
                            project_name: { type: 'text', fields: { keyword: { type: 'keyword' } } },
                            tracker_id: { type: 'keyword' }, // e.g., Bug, Feature
                            status_id: { type: 'keyword' }, // e.g., Open, Closed
                            priority_id: { type: 'keyword' },
                            author_id: { type: 'keyword' },
                            author_name: { type: 'text', fields: { keyword: { type: 'keyword' } } },
                            subject: { type: 'text', analyzer: 'ticket_analyzer' },
                            description: { type: 'text', analyzer: 'ticket_analyzer' },
                            category_id: { type: 'keyword' },
                            category_name: { type: 'text', fields: { keyword: { type: 'keyword' } } },
                            created_on: { type: 'date', format: 'yyyy-MM-dd HH:mm:ss||strict_date_optional_time||epoch_millis' },
                            updated_on: { type: 'date', format: 'yyyy-MM-dd HH:mm:ss||strict_date_optional_time||epoch_millis' },
                            file_path: { type: 'keyword' }, // For attachments
                            file_type: { type: 'keyword' },
                            attachment_content: { type: 'text', analyzer: 'ticket_analyzer' },
                            semantic_text: { type: 'text' }, // Combined text for semantic search
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
        model: 'text-embedding-3-large',
        input: text,
    });
    return response.data[0].embedding;
}

// Parse attachment content
async function parseAttachmentContent(filePath) {
    const fileExt = path.extname(filePath).toLowerCase();
    try {
        if (fileExt === '.pdf') {
            const dataBuffer = await fs.readFile(filePath);
            const pdfData = await pdfParse(dataBuffer);
            return pdfData.text;
        } else if (fileExt === '.docx') {
            const text = await docx2txt.extractText(filePath);
            return text;
        } else if (['.txt', '.md', '.json'].includes(fileExt)) {
            const content = await fs.readFile(filePath, 'utf-8');
            return fileExt === '.json' ? JSON.stringify(JSON.parse(content)) : content;
        }
        return '';
    } catch (e) {
        console.error(`Error parsing attachment ${filePath}:`, e);
        return '';
    }
}

// Retrieve Redmine document (issue, journal, or attachment)
async function retrieveRedmineDocument(docType, id, filePath) {
    if (docType === 'issue') {
        const issue = await prisma.issues.findUnique({
            where: { id: parseInt(id) },
            include: { projects: true, trackers: true, issue_statuses: true, enumerations: true, users: true },
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
            include: { users: true },
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
        const validPath = validateFilePath(filePath);
        if (!validPath) return null;

        const cacheKey = `attachment:${validPath}`;
        try {
            // Check Redis cache
            const cached = await redisClient.get(cacheKey);
            if (cached) {
                const { file_type, content } = JSON.parse(cached);
                return { file_type, content };
            }

            // Read and parse file
            const fileExt = path.extname(validPath).toLowerCase();
            const content = await parseAttachmentContent(validPath);
            const fileType = fileExt.slice(1); // e.g., 'pdf'

            // Cache in Redis
            await redisClient.setEx(cacheKey, CACHE_TTL, JSON.stringify({ file_type: fileType, content }));

            return { file_type: fileType, content };
        } catch (e) {
            console.error(`Error reading attachment ${validPath}:`, e);
            return null;
        }
    }
    return null;
}

// Validate file path
function validateFilePath(filePath) {
    try {
        const fullPath = path.resolve(UPLOAD_DIR, filePath);
        if (!SUPPORTED_FILE_EXTENSIONS.includes(path.extname(fullPath).toLowerCase())) {
            console.error(`Unsupported file extension: ${fullPath}`);
            return null;
        }
        return fullPath;
    } catch (e) {
        console.error(`Invalid file path ${filePath}:`, e);
        return null;
    }
}

// Generate dynamic OpenSearch query using GPT-4o
async function generateDynamicOpensearchQuery(query, context, chatHistory) {
    const functions = [
        {
            name: 'generate_opensearch_query',
            description: 'Generates an OpenSearch query for Redmine ticketing data retrieval.',
            parameters: {
                type: 'object',
                properties: {
                    intent: {
                        type: 'string',
                        enum: [
                            'SEMANTIC', 'KEYWORD', 'HYBRID', 'STRUCTURED', 'HYBRID_STRUCTURED',
                            'AGGREGATE', 'COMPARISON', 'TEMPORAL', 'EXPLANATORY', 'MULTI_INTENT',
                            'ENTITY_SPECIFIC', 'DOCUMENT_FETCH',
                        ],
                        description: 'The classified intent of the query.',
                    },
                    entities: {
                        type: 'array',
                        items: {
                            type: 'object',
                            properties: {
                                text: { type: 'string', description: 'The entity text.' },
                                label: {
                                    type: 'string',
                                    enum: [
                                        'ISSUE_ID', 'PROJECT_ID', 'PROJECT_NAME', 'TRACKER', 'STATUS',
                                        'PRIORITY', 'AUTHOR', 'CATEGORY', 'DATE', 'KEYWORD',
                                        'MEDICAL_TERM', 'ORGANIZATION', 'SOFTWARE_COMPONENT',
                                    ],
                                    description: 'The entity type.',
                                },
                            },
                            required: ['text', 'label'],
                        },
                        description: 'Extracted entities from the query.',
                    },
                    query_body: {
                        type: 'object',
                        description: 'The OpenSearch query body in JSON format.',
                        additionalProperties: true,
                    },
                },
                required: ['intent', 'entities', 'query_body'],
            },
        },
    ];

    const systemMsg = `
    You are an expert AI assistant for a Redmine ticketing system in medical informatics. Your task is to:
    1) Classify the query intent from: SEMANTIC, KEYWORD, HYBRID, STRUCTURED, HYBRID_STRUCTURED, AGGREGATE, COMPARISON, TEMPORAL, EXPLANATORY, MULTI_INTENT, ENTITY_SPECIFIC, DOCUMENT_FETCH.
    2) Extract entities (e.g., ISSUE_ID, PROJECT_NAME, MEDICAL_TERM) from the query.
    3) Generate a valid OpenSearch query_body for the classified intent, incorporating entities as filters.
    Rules:
    - Output only the JSON object returned by the 'generate_opensearch_query' function.
    - Use context and chat history to inform intent and entity extraction, but focus on the query.
    - For DOCUMENT_FETCH, prioritize issue_id or project_id filters.
    - For AGGREGATE, use aggregations (e.g., terms on status_id).
    - For SEMANTIC/HYBRID, include knn for embedding-based search on semantic_text.
    - Use terms lookup for lists exceeding ${MAX_TERMS} (e.g., issue_id, project_id).
    - Map medical terms (e.g., "billing issue") to synonyms via the ticket_analyzer.
    - Ensure query_body is valid OpenSearch DSL.
    - Example:
      Query: "Find tickets about billing issues in project X created in 2023"
      Output: {
        "intent": "HYBRID_STRUCTURED",
        "entities": [
          {"text": "billing issues", "label": "KEYWORD"},
          {"text": "X", "label": "PROJECT_NAME"},
          {"text": "2023", "label": "DATE"}
        ],
        "query_body": {
          "size": ${TOP_K},
          "query": {
            "bool": {
              "should": [
                {"knn": {"embedding": {"vector": [], "k": ${TOP_K}}}},
                {"match": {"semantic_text": "billing issues"}}
              ],
              "minimum_should_match": 1,
              "filter": [
                {"term": {"project_name.keyword": "X"}},
                {"range": {"created_on": {"gte": "2023-01-01", "lte": "2023-12-31"}}}
              ]
            }
          }
        }
      }
  `;

    const prompt = `
    Query: ${query}
    Chat History: ${chatHistory}
    Context: ${context}
    Generate the OpenSearch query using the 'generate_opensearch_query' function.
  `;

    const response = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
            { role: 'system', content: systemMsg },
            { role: 'user', content: prompt },
        ],
        functions: functions,
        function_call: { name: 'generate_opensearch_query' },
    });

    let result = JSON.parse(response.choices[0].message.function_call.arguments);

    // Handle large ID lists
    const issueIds = result.entities.filter(e => e.label === 'ISSUE_ID').map(e => e.text);
    const projectIds = result.entities.filter(e => e.label === 'PROJECT_ID').map(e => e.text);
    if (issueIds.length > MAX_TERMS) {
        const docId = await storeIdList(indexName, issueIds, 'issue_ids');
        result.query_body.query.bool.filter = result.query_body.query.bool.filter.filter(f => !f.terms?.issue_id);
        result.query_body.query.bool.filter.push({
            terms: { issue_id: { index: indexName, id: docId, path: 'issue_ids' } },
        });
    }
    if (projectIds.length > MAX_TERMS) {
        const docId = await storeIdList(indexName, projectIds, 'project_ids');
        result.query_body.query.bool.filter = result.query_body.query.bool.filter.filter(f => !f.terms?.project_id);
        result.query_body.query.bool.filter.push({
            terms: { project_id: { index: indexName, id: docId, path: 'project_ids' } },
        });
    }

    return result;
}

// Main ask function
async function ask(query, userId, chatId, topK = TOP_K) {
    if (!query.trim()) throw new Error('Empty query');
    if (!chatId || !userId) throw new Error('Missing user_id or chat_id');

    // Verify chat ownership
    const chat = await prisma.chat.findUnique({
        where: { id: chatId },
        include: { user: true },
    });
    if (!chat || chat.userId !== userId) {
        throw new Error('Chat not found or unauthorized');
    }

    const indexName = `${process.env.OPENSEARCH_INDEX_NAME || 'redmine'}-${userId}`;
    await ensureIndexExists(indexName);

    // Fetch chat history
    const messages = await prisma.message.findMany({
        where: { chatId },
        orderBy: { createdAt: 'desc' },
        take: MAX_CHAT_HISTORY,
    });
    const chatHistory = messages.reverse().map(m => `${m.role === 'user' ? 'User' : 'AI'}: ${m.content}`).join('\n');

    // Generate OpenSearch query
    const { intent, entities, query_body } = await generateDynamicOpensearchQuery(query, '', chatHistory);
    console.log(`Intent: ${intent}, Entities: ${JSON.stringify(entities)}, Query: ${JSON.stringify(query_body)}`);

    // Embed query for semantic/hybrid search
    const queryEmb = intent.includes('SEMANTIC') || intent.includes('HYBRID') || intent === 'MULTI_INTENT'
        ? await embedText(query)
        : null;
    if (queryEmb && query_body.query.bool?.should) {
        query_body.query.bool.should = query_body.query.bool.should.map(clause => {
            if (clause.knn) {
                return { knn: { embedding: { vector: queryEmb, k: topK } } };
            }
            return clause;
        });
    }

    // Execute search
    const results = await osClient.search({
        index: indexName,
        body: query_body,
        routing: entities.find(e => e.label === 'ISSUE_ID')?.text || entities.find(e => e.label === 'PROJECT_ID')?.text,
    });

    if (intent === 'AGGREGATE') {
        return JSON.stringify(results.aggregations || {}, null, 2);
    }

    // Retrieve documents
    const retrievedDocs = [];
    for (const hit of results.hits.hits) {
        const doc = hit._source;
        let docContent = null;
        if (doc.doc_type === 'issue') {
            docContent = await retrieveRedmineDocument('issue', doc.issue_id);
        } else if (doc.doc_type === 'journal') {
            docContent = await retrieveRedmineDocument('journal', doc.doc_id);
        } else if (doc.doc_type === 'attachment' && doc.file_path) {
            docContent = await retrieveRedmineDocument('attachment', null, doc.file_path);
        }
        if (docContent) {
            retrievedDocs.push({
                doc_id: doc.doc_id,
                doc_type: doc.doc_type,
                issue_id: doc.issue_id,
                project_id: doc.project_id,
                content: docContent.content,
                file_type: docContent.file_type,
            });
        }
    }

    if (!retrievedDocs.length) {
        return 'No relevant documents found.';
    }

    if (intent === 'DOCUMENT_FETCH') {
        return JSON.stringify({
            queried_entities: entities.map(e => ({ text: e.text, label: e.label })),
            matched_documents: retrievedDocs,
        }, null, 2);
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
            snippet = `[Attachment for Issue #${doc.issue_id}] Content: ${doc.content.slice(0, 200)}...`;
        }
        contextMap[docId] = contextMap[docId] ? `${contextMap[docId]}\n${snippet}` : snippet;
    });

    const contextText = Object.entries(contextMap)
        .map(([docId, content]) => `--- Document ID: ${docId} ---\n${content}\n\n`)
        .join('');

    // Generate final answer
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

    const answer = response.choices[0].message.content.trim();

    // Store messages
    const currentTime = new Date().toISOString();
    await prisma.message.createMany({
        data: [
            { chatId, role: 'user', content: query, createdAt: currentTime },
            { chatId, role: 'assistant', content: answer, createdAt: currentTime },
        ],
    });

    return answer;
}

// HTTP endpoint
app.post('/ask', async (req, res) => {
    try {
        const { query, user_id, chat_id, top_k = TOP_K } = req.body;
        if (!query || !user_id || !chat_id) {
            return res.status(400).json({ error: 'Provide user_id, chat_id, query' });
        }

        const answer = await ask(query, user_id, chat_id, top_k);
        res.json({ query, answer });
    } catch (e) {
        console.error('Error in /ask:', e);
        res.status(500).json({ error: e.message });
    }
});

// WebSocket endpoint
const wss = new WebSocketServer({ noServer: true });
wss.on('connection', async (ws) => {
    try {
        ws.on('message', async (data) => {
            try {
                const { query, user_id, chat_id, top_k = TOP_K } = JSON.parse(data);
                if (!query || !user_id || !chat_id) {
                    ws.send(JSON.stringify({ error: 'Missing required parameters.' }));
                    ws.close();
                    return;
                }

                // Verify chat ownership
                const chat = await prisma.chat.findUnique({
                    where: { id: chat_id },
                    include: { user: true },
                });
                if (!chat || chat.userId !== user_id) {
                    ws.send(JSON.stringify({ error: 'Chat not found or unauthorized access.' }));
                    ws.close();
                    return;
                }

                const indexName = `${process.env.OPENSEARCH_INDEX_NAME || 'redmine'}-${user_id}`;
                await ensureIndexExists(indexName);

                // Fetch chat history
                const messages = await prisma.message.findMany({
                    where: { chatId: chat_id },
                    orderBy: { createdAt: 'desc' },
                    take: MAX_CHAT_HISTORY,
                });
                const chatHistory = messages.reverse().map(m => `${m.role === 'user' ? 'User' : 'AI'}: ${m.content}`).join('\n');

                // Generate OpenSearch query
                const { intent, entities, query_body } = await generateDynamicOpensearchQuery(query, '', chatHistory);

                // Embed query
                const queryEmb = intent.includes('SEMANTIC') || intent.includes('HYBRID') || intent === 'MULTI_INTENT'
                    ? await embedText(query)
                    : null;
                if (queryEmb && query_body.query.bool?.should) {
                    query_body.query.bool.should = query_body.query.bool.should.map(clause => {
                        if (clause.knn) {
                            return { knn: { embedding: { vector: queryEmb, k: top_k } } };
                        }
                        return clause;
                    });
                }

                // Execute search
                const results = await osClient.search({
                    index: indexName,
                    body: query_body,
                    routing: entities.find(e => e.label === 'ISSUE_ID')?.text || entities.find(e => e.label === 'PROJECT_ID')?.text,
                });

                if (intent === 'AGGREGATE') {
                    const finalAnswer = JSON.stringify(results.aggregations || {}, null, 2);
                    ws.send(finalAnswer);

                    const currentTime = new Date().toISOString();
                    await prisma.message.createMany({
                        data: [
                            { chatId: chat_id, role: 'user', content: query, createdAt: currentTime },
                            { chatId: chat_id, role: 'assistant', content: finalAnswer, createdAt: currentTime },
                        ],
                    });
                    ws.close();
                    return;
                }

                // Retrieve documents
                const retrievedDocs = [];
                for (const hit of results.hits.hits) {
                    const doc = hit._source;
                    let docContent = null;
                    if (doc.doc_type === 'issue') {
                        docContent = await retrieveRedmineDocument('issue', doc.issue_id);
                    } else if (doc.doc_type === 'journal') {
                        docContent = await retrieveRedmineDocument('journal', doc.doc_id);
                    } else if (doc.doc_type === 'attachment' && doc.file_path) {
                        docContent = await retrieveRedmineDocument('attachment', null, doc.file_path);
                    }
                    if (docContent) {
                        retrievedDocs.push({
                            doc_id: doc.doc_id,
                            doc_type: doc.doc_type,
                            issue_id: doc.issue_id,
                            project_id: doc.project_id,
                            content: docContent.content,
                            file_type: docContent.file_type,
                        });
                    }
                }

                if (!retrievedDocs.length) {
                    ws.send(JSON.stringify({ answer: 'No relevant documents found.' }));
                    ws.close();
                    return;
                }

                if (intent === 'DOCUMENT_FETCH') {
                    const finalAnswer = JSON.stringify({
                        queried_entities: entities.map(e => ({ text: e.text, label: e.label })),
                        matched_documents: retrievedDocs,
                    }, null, 2);
                    ws.send(finalAnswer);

                    const currentTime = new Date().toISOString();
                    await prisma.message.createMany({
                        data: [
                            { chatId: chat_id, role: 'user', content: query, createdAt: currentTime },
                            { chatId: chat_id, role: 'assistant', content: finalAnswer, createdAt: currentTime },
                        ],
                    });
                    ws.close();
                    return;
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
                        snippet = `[Attachment for Issue #${doc.issue_id}] Content: ${doc.content.slice(0, 200)}...`;
                    }
                    contextMap[docId] = contextMap[docId] ? `${contextMap[docId]}\n${snippet}` : snippet;
                });

                const contextText = Object.entries(contextMap)
                    .map(([docId, content]) => `--- Document ID: ${docId} ---\n${content}\n\n`)
                    .join('');

                // Stream response
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

                const stream = await openai.chat.completions.create({
                    model: 'gpt-4o',
                    messages: [
                        { role: 'system', content: systemMsg },
                        { role: 'user', content: finalPrompt },
                    ],
                    max_tokens: 1024,
                    temperature: 0.7,
                    stream: true,
                });

                let finalAnswer = '';
                for await (const chunk of stream) {
                    const token = chunk.choices[0]?.delta?.content || '';
                    if (token) {
                        finalAnswer += token;
                        ws.send(token);
                    }
                }

                if (finalAnswer) {
                    const currentTime = new Date().toISOString();
                    await prisma.message.createMany({
                        data: [
                            { chatId: chat_id, role: 'user', content: query, createdAt: currentTime },
                            { chatId: chat_id, role: 'assistant', content: finalAnswer, createdAt: currentTime },
                        ],
                    });
                }

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
    await redisClient.quit();
    server.close(() => {
        console.log('Server closed.');
        process.exit(0);
    });
});