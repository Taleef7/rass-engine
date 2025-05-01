const DEFAULT_K = Number(process.env.DEFAULT_K || 25);
const EMBED_DIM = Number(process.env.EMBED_DIM || 1536);
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
        const filtered = all.filter(hit => !hit._score || hit._score >= 0.8);
        log(`[knnSearch] Hits after score filter (>=0.8): ${filtered.length}`);
        filtered.forEach(hit => log(`[knnSearch] Hit: id=${hit._id}, score=${hit._score}, text=${hit._source.text_chunk?.slice(0, 100)}...`));
        return filtered;
    } catch (err) {
        warn('knnSearch error:', err.message);
        return [];
    }
}

/**
 * Executes the ANN plan exactly as planned: 
 * one HNSW search per 'search_term' in the plan.
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

    const bestById = new Map();                // _id → best-scoring hit

    for (const step of plan) {
        console.log(`[runSteps] Processing step: ${JSON.stringify(step)}`);
        const term = step.search_term?.trim();          // single term
        const vector = await embedOnce(term);            // embed as-is

        const k = step.knn_k || DEFAULT_K;
        const body = {
            size: k,
            _source: ['doc_id', 'file_path', 'file_type', 'text_chunk'],
            query: { knn: { embedding: { vector, k } } }
        };

        const { body: res } = await knnSearch(os, index, body);
        console.log("Result=", res);
        for (const h of res?.hits?.hits || []) {
            const prev = bestById.get(h._id);
            if (!prev || h._score > prev._score) bestById.set(h._id, h);
        }
    }

    return Array.from(bestById.values());
}


module.exports = { runSteps };