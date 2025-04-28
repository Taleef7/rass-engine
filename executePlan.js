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

async function runPlan(
    plan,
    queryText,
    indexName,
    mappings,
    osClient,
    embedText,
) {
    const results = new Map();
    const queryEmb = {};                              // cache by step

    for (const step of plan) {
        const {
            step_id,
            query_body,
            purpose = 'unspecified',
            requires_embedding,
            depends_on = [],
        } = step;

        /* Make sure all dependencies have been produced */
        const depsMissing = depends_on.some(d => !results.has(d));
        if (depsMissing) {
            console.warn(`Skipping ${step_id} – unmet dependency`);
            continue;
        }

        /* Inject embedding only ONCE */
        if (requires_embedding) {
            if (!queryEmb.vector) queryEmb.vector = await embedText(queryText);
            traverseAndReplaceKnn(query_body, queryEmb.vector);
        }

        /* Inject ids from previous steps for chained terms */
        injectChainedIds(query_body, depends_on, results, indexName, osClient);

        /* Schema validation */
        validateOpensearchQuery(query_body, mappings);

        /* Scroll search */
        const allHits = await gatherAllHits(osClient, indexName, query_body);

        results.set(step_id, { purpose, hits: allHits });
    }

    return results;
}

/* --------------------------- helpers ------------------------------ */

function traverseAndReplaceKnn(obj, vector) {
    if (obj?.knn?.embedding) obj.knn.embedding.vector = vector;
    for (const k in obj) if (typeof obj[k] === 'object') traverseAndReplaceKnn(obj[k], vector);
}

async function injectChainedIds(queryBody, deps, results, indexName, osClient) {
    const idSet = new Set();
    for (const d of deps) {
        for (const h of results.get(d).hits) idSet.add(h._source.issue_id);
    }
    if (!idSet.size) return;

    const ids = [...idSet];
    const termsNode = {
        terms: ids.length > 1000
            ? { issue_id: { index: indexName, id: await storeIdList(indexName, ids, 'issue_ids'), path: 'issue_ids' } }
            : { issue_id: ids },
    };

    // plumb into the filter array
    const bool = queryBody.query?.bool;
    if (bool) {
        bool.filter = Array.isArray(bool.filter) ? bool.filter : [];
        bool.filter.push(termsNode);
    }
}

async function gatherAllHits(osClient, indexName, body) {
    const hits = [];
    const first = await osClient.search({ index: indexName, body, scroll: '90s' });
    hits.push(...first.hits.hits);

    let sid = first._scroll_id;
    while (true) {
        const nxt = await osClient.scroll({ scroll_id: sid, scroll: '90s' });
        if (!nxt.hits.hits.length) break;
        hits.push(...nxt.hits.hits);
        sid = nxt._scroll_id;
    }
    await osClient.clearScroll({ scroll_id: [sid] });
    return hits;
}

module.exports = runPlan;
