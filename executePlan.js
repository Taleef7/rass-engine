/**********************************************************************
 * executePlan.js  – run an OpenSearch plan with automatic k-NN recall
 *********************************************************************/
const { v4: uuidv4 } = require('uuid');

const EMBED_DIM = Number(process.env.EMBED_DIM || 1536);
const DEFAULT_K = Number(process.env.DEFAULT_K || 25);

const VALID_FIELDS = new Set([
    'doc_id', 'doc_type', 'issue_id',
    'file_path', 'file_type',
    'attachment_content', 'text_chunk',
    'embedding'
]);

/* ───────────────── utilities ───────────────── */
function log(...a) { console.log(...a); }
function warn(...a) { console.warn(...a); }

async function storeIdList(os, index, ids, field) {
    const id = `${field}_list_${uuidv4()}`;
    await os.index({ index, id, body: { [field]: ids } });
    return id;
}

function validate(body) {
    const walk = o => {
        for (const k in o) {
            if (k === 'knn' && o[k].embedding) {
                const v = o[k].embedding.vector || [];
                if (v.length !== EMBED_DIM)
                    throw Error(`knn vector length ${v.length} ≠ ${EMBED_DIM}`);
            } else if (['match', 'term', 'terms', 'range'].includes(k)) {
                const f = Object.keys(o[k])[0]?.split('.')[0];
                if (!VALID_FIELDS.has(f))
                    warn(`⚠️ unknown field “${f}” (will still send)`);
            } else if (o[k] && typeof o[k] === 'object') walk(o[k]);
        }
    };
    walk(body); return true;
}

function injectVector(body, vector, k) {
    body.query = { knn: { embedding: { vector, k } } };
    body.size = body.size || k;
}

function replaceVectors(obj, vec, k) {
    if (obj?.knn?.embedding) {
        obj.knn.embedding.vector = vec;
        obj.knn.embedding.k = obj.knn.embedding.k || k;
    }
    for (const key in obj)
        if (obj[key] && typeof obj[key] === 'object')
            replaceVectors(obj[key], vec, k);
}

async function injectChainedIds(body, deps, res, index, os) {
    const set = new Set();
    for (const d of deps) for (const h of res.get(d)?.hits || [])
        set.add(h._source.issue_id);
    if (!set.size) return;

    const ids = [...set];
    const clause = ids.length > 1e3
        ? { terms: { issue_id: { index, id: await storeIdList(os, index, ids, 'issue_ids'), path: 'issue_ids' } } }
        : { terms: { issue_id: ids } };

    body.query.bool = body.query.bool || {};
    body.query.bool.filter = [].concat(body.query.bool.filter || [], clause);
}

async function scrollAll(os, index, body) {
    const hits = [];
    let resp = await os.search({ index, body, scroll: '60s' });
    let sid = resp.body._scroll_id;
    hits.push(...(resp.body.hits.hits || []));

    while (true) {
        resp = await os.scroll({ scroll_id: sid, scroll: '60s' });
        if (!resp.body.hits.hits.length) break;
        hits.push(...resp.body.hits.hits);
        sid = resp.body._scroll_id;
    }
    await os.clearScroll({ scroll_id: [sid] });
    return hits;
}

/* ───────────────── main export ───────────────── */
async function runPlan(plan, queryText, indexName, mappings, osClient, embedText) {
    const results = new Map();
    let queryVec = null;

    /* ------------------------------------------------------------------
     * Ensure we have a semantic step first.  If the planner already
     * provided one (requires_embedding:true) we keep the order; otherwise
     * we prepend our own.
     * ------------------------------------------------------------------ */
    const hasKnn = plan.some(s => s.requires_embedding || s.requiresEmbedding);
    const fullPlan = hasKnn ? plan : [
        {
            step_id: 'semantic_recall',
            purpose: 'initial semantic similarity recall',
            requires_embedding: true,
            depends_on: [],
            query_body: {
                query: { knn: { embedding: { vector: [], k: DEFAULT_K } } },
                _source: ['doc_id', 'file_path'],
                size: DEFAULT_K
            },
            is_final: false
        },
        ...plan
    ];

    /* ------------------------------------------------------------------ */
    for (const step of fullPlan) {
        log('\n▶️ [runPlan] step raw:', JSON.stringify(step, null, 2));

        const id = step.step_id || step.stepId || 'step?';
        const deps = step.depends_on ?? step.dependsOn ?? [];
        const wantsVec = step.requires_embedding ?? step.requiresEmbedding;
        let body = step.query_body || step.queryBody;
        const purpose = step.purpose || 'unspecified';

        if (!body) { warn(`⚠️ ${id} missing query_body – skipped`); continue; }
        if (deps.some(d => !results.has(d))) { warn(`⚠️ ${id} deps missing – skipped`); continue; }

        const topK = body.size || DEFAULT_K;
        if (!body.size) body.size = topK;

        /* auto-vector if requested */
        if (wantsVec) {
            if (!queryVec) queryVec = await embedText(queryText);
            replaceVectors(body, queryVec, topK);
        }

        /* if planner gave a term/match on unknown field → vector fallback */
        if (!wantsVec) {
            const q = body.query || {};
            for (const kw of ['term', 'match', 'terms']) {
                if (q[kw]) {
                    const f = Object.keys(q[kw])[0]?.split('.')[0];
                    if (f && !VALID_FIELDS.has(f)) {
                        warn(`⚠️ unknown field “${f}” → automatic semantic recall`);
                        if (!queryVec) queryVec = await embedText(queryText);
                        injectVector(body, queryVec, topK);
                    }
                }
            }
        }

        /* chained ids */
        await injectChainedIds(body, deps, results, indexName, osClient);

        /* final checks + log */
        validate(body);
        log('▶️ [runPlan] Step', id, 'DSL:', JSON.stringify(body, null, 2));

        /* execute */
        const hits = await scrollAll(osClient, indexName, body);
        results.set(id, { purpose, hits });
    }

    return results;          // may contain zero-hit steps – caller handles
}

module.exports = runPlan;
