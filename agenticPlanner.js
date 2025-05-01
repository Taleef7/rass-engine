/**********************************************************************
 * agenticPlanner.js – robust planner for OpenSearch RAG
 *  • Uses OpenAI to generate multi-step DSL plans
 *  • Auto-repairs empty / bad plans with smart fallbacks
 *********************************************************************/
const MAX_AGENT_ITER = 6;
const MAX_PLAN_STEPS = 15;

const PLAN_GEN_FN = 'generate_query_plan';
const COVERAGE_FN = 'evaluate_coverage';
const DEFAULT_K = Number(process.env.DEFAULT_K || 25);

/* ------------ helper to create a semantic knn step --------------- */
function semanticStep(stepId, k = DEFAULT_K) {
    return {
        step_id: stepId,
        purpose: 'semantic similarity fallback',
        requires_embedding: true,
        depends_on: [],
        is_final: true,
        query_body: {
            _source: ['doc_id', 'file_path', 'attachment_content'],
            size: k,
            query: {
                knn: {
                    embedding: { vector: [], k }
                }
            }
        }
    };
}

/* ---------- helper to create a fuzzy / prefix text step ---------- */
function prefixStep(stepId, queryText, k = DEFAULT_K) {
    return {
        step_id: stepId,
        purpose: 'fuzzy / prefix text fallback',
        requires_embedding: false,
        depends_on: [],
        is_final: false,
        query_body: {
            _source: ['doc_id', 'file_path', 'attachment_content'],
            size: k,
            query: {
                bool: {
                    should: [
                        { match_phrase_prefix: { attachment_content: queryText } },
                        { wildcard: { attachment_content: `${queryText}*` } }
                    ]
                }
            }
        }
    };
}

/* ----------------------------- main loop ------------------------- */
async function planAndExecute({
    query,
    openai,
    osClient,
    indexName,
    mappings,
    embedText,
    runPlan
}) {
    const history = [];
    let iteration = 0;

    while (iteration++ < MAX_AGENT_ITER) {

        /* ---- 1. get plan from the LLM (may fail) ------------------- */
        let plan = [];
        try {
            ({ plan } = await getPlanFromLLM({ openai, query, history }));
        } catch (e) {
            console.log('[planner] LLM returned no plan – building fallback.');
        }

        /* ---- 2. Fallback if plan empty or invalid ------------------ */
        if (!Array.isArray(plan) || plan.length === 0) {
            plan = [
                prefixStep('text_prefix', query, DEFAULT_K),
                semanticStep('semantic_fallback', DEFAULT_K)
            ];
        }

        if (plan.length > MAX_PLAN_STEPS) {
            plan = plan.slice(0, MAX_PLAN_STEPS);
        }

        /* ---- 3. Normalise and sanity-check each step --------------- */
        for (const step of plan) {
            // allow camelCase from LLM
            if (step.queryBody && !step.query_body) {
                step.query_body = step.queryBody;
                delete step.queryBody;
            }
            if (!step.query_body) {
                // if still missing, replace with semantic step
                Object.assign(step, semanticStep(step.step_id || 'auto_semantic'));
            }
            // guarantee numeric size
            if (step.query_body.size == null) step.query_body.size = DEFAULT_K;
        }

        /* ---- 4. Execute -------------------------------------------- */
        const execResults = await runPlan(
            plan,
            query,
            indexName,
            mappings,
            osClient,
            embedText
        );

        /* ---- 5. summarise for next round --------------------------- */
        const summary = [...execResults.entries()].map(([id, r]) => ({
            step_id: id,
            hit_count: r.hits.length
        }));
        history.push(...summary);

        /* stop if nothing at all was found */
        if (summary.every(s => s.hit_count === 0)) {
            return execResults;
        }

        /* ask the LLM if coverage is complete ------------------------ */
        const covered = await checkCoverage({
            openai,
            query,
            history: summary
        });
        if (covered) return execResults;
    }

    console.log(
        `[planner] gave up after ${MAX_AGENT_ITER} iterations – returning what we have`
    );
    return new Map(); // may be empty
}

/* --------------- LLM call to produce the plan ------------------- */
async function getPlanFromLLM({ openai, query, history }) {
    const tools = [
        {
            type: 'function',
            function: {
                name: PLAN_GEN_FN,
                description:
                    'Return only JSON: { "plan":[ {step_id, query_body, purpose, requires_embedding, depends_on, is_final} ] }',
                parameters: {
                    type: 'object',
                    properties: {
                        plan: { type: 'array', items: { type: 'object' } }
                    },
                    required: ['plan']
                }
            }
        }
    ];

    const resp = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
            {
                role: 'system',
                content:
                    'You are building OpenSearch DSL plans. Output ONLY JSON for { "plan": [...] }.'
            },
            {
                role: 'user',
                content: `User query: ${query}\nPrevious summary: ${JSON.stringify(
                    history
                )}`
            }
        ],
        tools,
        tool_choice: { type: 'function', function: { name: PLAN_GEN_FN } }
    });

    const call = resp.choices?.[0]?.message?.tool_calls?.[0];
    if (!call?.function?.arguments) return { plan: [] };

    console.log('[planner] raw function arguments:', call.function.arguments);

    try {
        return JSON.parse(call.function.arguments);
    } catch {
        return { plan: [] };
    }
}

/* ---------------- coverage yes/no helper ------------------------ */
async function checkCoverage({ openai, query, history }) {
    const tools = [
        {
            type: 'function',
            function: {
                name: COVERAGE_FN,
                description: 'Return { "covered": boolean }',
                parameters: {
                    type: 'object',
                    properties: { covered: { type: 'boolean' } },
                    required: ['covered']
                }
            }
        }
    ];

    try {
        const resp = await openai.chat.completions.create({
            model: 'gpt-4o',
            messages: [
                { role: 'system', content: 'Answer with function call only.' },
                {
                    role: 'user',
                    content: `Query: ${query}\nHit summary: ${JSON.stringify(history)}`
                }
            ],
            tools,
            tool_choice: { type: 'function', function: { name: COVERAGE_FN } }
        });

        return JSON.parse(
            resp.choices[0].message.tool_calls[0].function.arguments
        ).covered;
    } catch {
        return false;
    }
}

module.exports = { planAndExecute };
