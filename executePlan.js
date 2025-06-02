const DEFAULT_K = Number(process.env.DEFAULT_K || 25);
const EMBED_DIM = Number(process.env.EMBED_DIM || 1536);
const envScoreThreshold = parseFloat(process.env.OPENSEARCH_SCORE_THRESHOLD);
const OPENSEARCH_SCORE_THRESHOLD = !isNaN(envScoreThreshold) ? envScoreThreshold : 0.78; // More robust check
const SCROLL_TTL = '60s';

const log = (...a) => console.log(...a);
const warn = (...a) => console.warn(...a);

/**
 * Runs a single knn search with scroll.
 */
async function knnSearch(os, index, body) {
    try {
        const first = await os.search({ index, body, scroll: SCROLL_TTL });
        let sid = first.body._scroll_id;
        const all = [...first.body.hits.hits];

        while (true) {
            const nxt = await os.scroll({ scroll_id: sid, scroll: SCROLL_TTL });
            if (!nxt.body.hits.hits.length) break;
            all.push(...nxt.body.hits.hits);
            sid = nxt.body._scroll_id;
        }
        await os.clearScroll({ scroll_id: [sid] });
        // === MODIFIED LINES START ===
        const filtered = all.filter(hit => !hit._score || hit._score >= OPENSEARCH_SCORE_THRESHOLD);
        log(`[knnSearch] Hits after score filter (>=${OPENSEARCH_SCORE_THRESHOLD}): ${filtered.length}`);
        // === MODIFIED LINES END ===
        filtered.forEach(hit => log(`[knnSearch] Hit: id=${hit._id}, score=${hit._score}`));
        return filtered;
    } catch (err) {
        warn('knnSearch error:', err.message);
        return [];
    }
}

/**
 * Executes the ANN plan exactly as planned:
 *  - run one HNSW search per 'search_term'
 *  - keep each step’s hits in descending score order
 *  - then interleave: step1[0], step2[0], ..., stepN[0], step1[1], step2[1], ...
 *  - dedupe by _id, preserving first appearance
 */
async function runSteps({ plan, embed, os, index }) {
    const VEC_CACHE = new Map();               // text → embedding

    async function embedOnce(text) {
        if (!VEC_CACHE.has(text)) {
            const vec = await embed(text);
            if (!Array.isArray(vec) || vec.length !== EMBED_DIM)
                throw new Error(`Bad embedding length for "${text}"`);
            VEC_CACHE.set(text, vec);
        }
        return VEC_CACHE.get(text);
    }

    // collect each step’s raw hits
    const perStepHits = [];
    for (const step of plan) {
        // console.log(`[runSteps] Processing step: ${JSON.stringify(step)}`);
        const term = step.search_term?.trim();
        const vector = await embedOnce(term);
        const k = step.knn_k || DEFAULT_K;
        const body = {
            size: k,
            _source: ['doc_id', 'file_path', 'file_type', 'text_chunk'],
            query: { knn: { embedding: { vector, k } } }
        };

        const hits = await knnSearch(os, index, body);
        perStepHits.push(hits);
    }

    // interleave them
    const interleaved = [];
    const seen = new Set();
    const maxLen = Math.max(...perStepHits.map(h => h.length), 0);

    for (let i = 0; i < maxLen; i++) {
        for (const hits of perStepHits) {
            const hit = hits[i];
            if (hit && !seen.has(hit._id)) {
                seen.add(hit._id);
                interleaved.push(hit);
            }
        }
    }

    return interleaved;
}


module.exports = { runSteps };