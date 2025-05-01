/**********************************************************************
 * agenticPlanner.js – resilient OpenAI-driven planner
 * - Auto-repairs steps missing `query_body`
 * - Synthesises a fallback semantic step when the LLM returns nothing
 * - Never throws on planner defects; caller just gets an empty docs list
 *********************************************************************/
const MAX_AGENT_ITER = 6;
const MAX_PLAN_STEPS = 15;
const PLAN_GEN_FN = 'generate_query_plan';
const COVERAGE_FN = 'evaluate_coverage';

const DEFAULT_K = Number(process.env.DEFAULT_K || 25);

/* ----------------------------- tiny log helpers ------------------ */
const log = (...a) => console.log(...a);
const warn = (...a) => console.warn(...a);

/* ------------------------ local builders ------------------------- */
function makeSemanticStep(stepId, userQuery) {
    return {
        step_id: stepId,
        purpose: 'semantic similarity fallback',
        requires_embedding: true,
        depends_on: [],
        is_final: true,
        query_body: {
            _source: ['doc_id', 'file_path', 'attachment_content'],
            size: DEFAULT_K,
            query: {
                knn: { embedding: { vector: [], k: DEFAULT_K } }
            }
        }
    };
}

function makeDocIdStep(stepId, docId) {
    return {
        step_id: stepId,
        purpose: `exact match on doc_id = ${docId}`,
        requires_embedding: false,
        depends_on: [],
        is_final: true,
        query_body: {
            _source: ['*'],
            query: { term: { doc_id: docId } },
            size: 25
        }
    };
}

function guessDocId(text) {
    const m = /\b([A-Z]?[a-z0-9_-]{3,})\b/i.exec(text);
    return m ? m[1] : null;
}

/* ------------------------------- main loop ----------------------- */
async function planAndExecute({
    query, openai, osClient,
    indexName, mappings,
    embedText, runPlan
}) {
    const history = [];
    let iter = 0;

    while (++iter <= MAX_AGENT_ITER) {

        /* ----- 1) ask LLM for a plan (may fail) ---------------------- */
        let plan = [];
        try {
            ({ plan } = await getPlanFromLLM({ openai, query, history }));
        } catch (e) {
            warn(`⚠️  LLM planning error (${e.message}); falling back to semantic step`);
        }

        /* ----- 2) build fallback if planner useless ----------------- */
        if (!Array.isArray(plan) || !plan.length) {
            plan = [makeSemanticStep('semantic_fallback', query)];
            warn('⚠️  using auto-generated fallback plan');
        }

        /* guard-rails                                                  */
        if (plan.length > MAX_PLAN_STEPS) {
            warn(`⚠️  truncating plan to first ${MAX_PLAN_STEPS} steps`);
            plan = plan.slice(0, MAX_PLAN_STEPS);
        }

        /* ----- 3) repair each step that lacks query_body ------------ */
        for (const step of plan) {
            if (step.query_body || step.queryBody) continue;

            warn(`⚠️  step ${step.step_id || step.stepId} missing query_body – repairing`);
            const docId = guessDocId(step.purpose || '') || guessDocId(query);
            Object.assign(
                step,
                docId ? makeDocIdStep(step.step_id || step.stepId, docId)
                    : makeSemanticStep(step.step_id || step.stepId, query)
            );
        }

        /* camelCase → snake_case                                       */
        for (const s of plan) {
            if (s.queryBody && !s.query_body) {
                s.query_body = s.queryBody;
                delete s.queryBody;
            }
        }

        /* ----- 4) execute ------------------------------------------- */
        const exec = await runPlan(
            plan, query, indexName, mappings, osClient, embedText
        );

        /* hit summary for the next LLM turn                            */
        const summary = [...exec.entries()].map(([id, r]) => ({
            step_id: id, hit_count: r.hits.length
        }));
        history.push(...summary);

        /* stop if everything empty or coverage says we’re done         */
        if (summary.every(s => s.hit_count === 0)) return exec;
        const covered = await checkCoverage({ openai, query, history: summary });
        if (covered) return exec;
    }

    warn(`Planner gave up after ${MAX_AGENT_ITER} iterations`);
    return new Map();                 // empty result map
}

/* -------------------- helper: call the LLM ---------------------- */
async function getPlanFromLLM({ openai, query, history }) {
    const tools = [{
        type: 'function',
        function: {
            name: PLAN_GEN_FN,
            description: 'Return { "plan":[ … ] }',
            parameters: {
                type: 'object',
                properties: { plan: { type: 'array', items: { type: 'object' } } },
                required: ['plan']
            }
        }
    }];

    const resp = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
            {
                role: 'system', content:
                    'Return ONLY JSON for { "plan":[ … ] }. No text.'
            },
            {
                role: 'user',
                content: `User query: ${query}\nHistory: ${JSON.stringify(history)}`
            }
        ],
        tools,
        tool_choice: { type: 'function', function: { name: PLAN_GEN_FN } }
    });

    const call = resp.choices?.[0]?.message?.tool_calls?.[0];
    if (!call || !call.function || !call.function.arguments)
        throw Error('LLM returned no tool arguments');

    log('▶️ [planner] raw function arguments:', call.function.arguments);

    let parsed;
    try { parsed = JSON.parse(call.function.arguments); }
    catch { throw Error('invalid JSON from planner'); }

    return parsed;
}

/* --------------- helper: quick coverage check ------------------- */
async function checkCoverage({ openai, query, history }) {
    const tools = [{
        type: 'function',
        function: {
            name: COVERAGE_FN,
            description: 'Return { "covered": boolean }',
            parameters: {
                type: 'object',
                properties: { covered: { type: 'boolean' } }, required: ['covered']
            }
        }
    }];

    try {
        const resp = await openai.chat.completions.create({
            model: 'gpt-4o',
            messages: [
                { role: 'system', content: 'Answer with function call only.' },
                {
                    role: 'user', content:
                        `Query: ${query}\nHit summary: ${JSON.stringify(history)}`
                }
            ],
            tools,
            tool_choice: { type: 'function', function: { name: COVERAGE_FN } }
        });

        return JSON.parse(resp.choices[0].message.tool_calls[0]
            .function.arguments).covered;
    } catch {
        return false;   // if the check fails we keep iterating
    }
}

module.exports = { planAndExecute };
