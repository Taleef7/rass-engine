const MAX_AGENT_ITER = 6;                 // hard stop ⟵ safety-valve
const MAX_PLAN_STEPS = 15;                // protect the scroll API
const COVERAGE_FN = 'evaluate_coverage';  // name of the reflection tool
const PLAN_GEN_FN = 'generate_query_plan';  // name of the planning tool

/**
 * High-level agentic loop.
 * Keeps proposing plans → executes → reflects → (maybe) revises
 */
async function planAndExecute({
    query,
    openai,
    osClient,
    indexName,
    mappings,
    embedText,
    runPlan,
}) {
    const history = [];               // [{step_id, hits:[…]}, …]
    let iteration = 0;

    while (iteration++ < MAX_AGENT_ITER) {
        /* Ask GPT-4o for a fresh / revised plan */
        const { plan } = await getPlanFromLLM({
            openai,
            query,
            history,
            indexName,
        });

        /* Guardrails */
        if (!Array.isArray(plan) || plan.length === 0) {
            throw new Error('Planner returned an empty plan');
        }
        if (plan.length > MAX_PLAN_STEPS) {
            throw new Error(`Planner exploded to ${plan.length} steps`);
        }

        /* Run the plan against OpenSearch */
        const execResults = await runPlan(plan, query, indexName, mappings, osClient, embedText);

        /* Append summary to history that the next LLM call can see */
        const summary = [...execResults.entries()].map(([step_id, r]) => ({
            step_id,
            hit_count: r.hits.length,
            terms: new Set(r.hits.flatMap(h => Object.values(h._source))),
        }));
        history.push(...summary);

        /* Ask the LLM if we are DONE */
        const covered = await checkCoverage({
            openai,
            query,
            history: summary,
        });
        if (covered) {
            return execResults;           // success
        }
    }

    throw new Error(
        `Agent exited after ${MAX_AGENT_ITER} iterations – query may still be incomplete`,
    );
}

/* --------------------------- helpers ------------------------------ */

async function getPlanFromLLM({ openai, query, history }) {
    const tools = [
        {
            type: 'function',
            function: {
                name: PLAN_GEN_FN,
                description:
                    'Returns a JSON object with a field "plan": an array of OpenSearch query steps.',
                parameters: {
                    type: 'object',
                    properties: {
                        plan: {
                            type: 'array',
                            items: {
                                type: 'object',
                                properties: {
                                    step_id: { type: 'string' },
                                    query_body: { type: 'object' },
                                    purpose: { type: 'string' },
                                    requires_embedding: { type: 'boolean' },
                                    depends_on: { type: 'array', items: { type: 'string' } },
                                },
                                required: ['step_id', 'query_body', 'purpose', 'requires_embedding', 'depends_on'],
                            },
                        },
                    },
                    required: ['plan'],
                },
            },
        },
    ];

    const resp = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
            {
                role: 'system',
                content: `You are the planner for a medical-document RAG engine backed by OpenSearch.
Create the *minimal* sequence of query steps that will satisfy the user request.
Use term filters, knn vectors, and chaining (terms lookup) as needed.
Return ONLY the JSON for generate_query_plan.`,
            },
            { role: 'user', content: `User query: ${query}\nPrevious rounds: ${JSON.stringify(history)}` },
        ],
        tools,
        tool_choice: { type: 'function', function: { name: 'generate_query_plan' } },
    });

    return JSON.parse(resp.choices[0].message.tool_calls[0].function.arguments);
}

async function checkCoverage({ openai, query, history }) {
    const tools = [
        {
            type: 'function',
            function: {
                name: COVERAGE_FN,
                description:
                    'Returns { "covered": true|false } indicating whether the fetched results fully answer the query.',
                parameters: {
                    type: 'object',
                    properties: {
                        covered: { type: 'boolean' },
                    },
                    required: ['covered'],
                },
            },
        },
    ];

    const resp = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
            {
                role: 'system',
                content: `You are an expert validator. Decide if the retrieved hits fully satisfy the request.`,
            },
            {
                role: 'user',
                content: `Original query: ${query}\nResult summary: ${JSON.stringify(history)}`,
            },
        ],
        tools,
        tool_choice: { type: 'function', function: { name: COVERAGE_FN } },
    });

    return JSON.parse(resp.choices[0].message.tool_calls[0].function.arguments)?.covered;
}

module.exports = {
    planAndExecute,
};
