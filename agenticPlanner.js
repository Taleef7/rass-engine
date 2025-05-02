const MAX_ITER = 6;
const DEFAULT_K = Number(process.env.DEFAULT_K || 25);
const EMBED_DIM = Number(process.env.EMBED_DIM || 1536);


/**
 * Convert a natural-language request into an ANN-only plan
 * understood by `runSteps`.
 *
 * Output schema:
 * {
 *   intent:   string[],
 *   entities: [{ text, type }],
 *   expansions: { <entity>: string[] },
 *   plan: [{ step_id, search_term, knn_k }]
 * }
 */
async function buildPlan(openai, query, history = []) {
  const sysPrompt = `
You are an expert OpenSearch strategist.  
The index schema is **immutable**:

mappings: {
  properties: {
    doc_id     : { type: keyword },
    file_path  : { type: keyword },
    file_type  : { type: keyword },
    text_chunk : { type: text    },
    embedding  : { type: knn_vector, dimension: ${EMBED_DIM},
                   method: { name: hnsw, engine: nmslib,
                             space_type: cosinesimil,
                             parameters: { m: 48, ef_construction: 400 } } }
  }
}

### Your job
1.  Detect the user **intent** (one short phrase).
2.  Extract **named entities** – label each as
    PERSON/PROPER NOUN, DATE, PLACE, MEDICAL_COND, ORG, ID, OTHER_TERM.
3.  Produce up to three concise **expansions** per entity
    (skip people, IDs, or ultra-specific proper nouns).
4.  Craft an ANN-only **plan**:
    • One 'search_term' per step (entity or expansion).  
    • Choose 'knn_k' (10 for specific, up to 50 for broad).

### Response
Return **ONLY** valid JSON (no markdown). Use the schema above.

### Examples
(abridged for brevity – note the 'search_term' key and look at the first example for the JSON output structure)

1. Query: *"get me records for Juli and the documents having the mention of terms containing Borne"*  
   → intent: "find documents"
   → entities: [{"text": "Juli", "type": "PERSON/PROPER NOUN"}, {"text": "Borne", "type": "PERSON/PROPER NOUN"}]
   → expansions: {"Juli": [], "Borne": []} 
   → **Your Output**:
     {
       "intent": ["find documents"],
       "plan": [
         {
           "step_id": "e1",
           "search_term": "Juli",
           "knn_k": 10,
         },
         {
           "step_id": "e2",
           "search_term": "Borne",
           "knn_k": 10,
         }       
        ]
      }.

2. Query: *"heart disease reports"*
   → intent: ["find documents"]  
   → entities: [{"text":"heart disease","type":"MEDICAL_COND"}]  
   → expansions: {"heart disease":["cardiac disorder","cardiovascular disease"]}  
   → plan: steps e1-e3 over those three terms, k=25.
   → **Your Output**:
     {
       "intent": ["find documents"],
       "plan": [
         {
           "step_id": "e1",
           "search_term": "heart disease",
           "knn_k": 10,
         },
         {
           "step_id": "e2",
           "search_term": "cardiac disorder",
           "knn_k": 10,
         },
         ... cover all the expansions and entities like this ...       
        ]
      }.

3. Query: *"COVID-19 cases in Seattle 2021"*  
   → entities: COVID-19 (MEDICAL_COND), Seattle (PLACE), 2021 (DATE).  
   → expansions only for COVID-19. Three steps, k=20/10/10.
   Your output should again look like the example one and two's JSON.

4. Query: *"Memorial Sloan Kettering diabetes 2023"*  
   → entities ORG+MEDICAL_COND+DATE, expansions for diabetes. Five steps.
   Your output should again look like the example one and two's JSON.

5. Query: *"patients with hypertension and asthma in California after June 2024"*  
   → hypertension, asthma, California, June 2024 (+ expansions). Six steps.
   Your output should again look like the example one and two's JSON.

6. Query: *"research on Alzheimer’s biomarkers"*  
   → entities: Alzheimer’s (MEDICAL_COND), biomarkers (OTHER_TERM)  
   → expansions for Alzheimer’s only. Two steps, k=25 each.
   Your output should again look like the example one and two's JSON.

Current Query: ${query}
History: ${JSON.stringify(history)}
`.trim();

  const resp = await openai.chat.completions.create({
    model: 'gpt-4o',
    temperature: 0.3,
    max_tokens: 1500,
    messages: [
      { role: 'system', content: sysPrompt },
      { role: 'user', content: `Query: ${query}\nHistory: ${JSON.stringify(history)}` }
    ]
  });

  try {
    return JSON.parse(resp?.choices[0]?.message?.content);
  } catch (e) {
    console.warn('[LLM] bad JSON, falling back:', e.message);
    return {                                // emergency single-step plan
      intent: ["find documents"],
      entities: [{ text: query, type: "OTHER_TERM" }],
      expansions: { [query]: [] },
      plan: [{ step_id: "e1", search_term: query, knn_k: DEFAULT_K, is_final: true }]
    };
  }
}


/**
 * REPL-style loop: build → run → decide if done.
 * 'mappings' is optional; if supplied, we verify that the index
 * matches the canonical schema once and warn otherwise.
 */
async function planAndExecute({
  query,
  openai,
  osClient,
  indexName,
  mappings = null,       // optional
  embedText,
  runStepsFn   // injectable for tests
}) {
  if (mappings) {
    const allowed = ['doc_id', 'file_path', 'file_type', 'text_chunk', 'embedding'];
    const bad = Object.keys(mappings.properties || {})
      .filter(f => !allowed.includes(f));
    if (bad.length) console.warn('[planAndExecute] unmapped fields:', bad);
  }

  const history = [];
  for (let iter = 0; iter < MAX_ITER; ++iter) {
    const planObj = await buildPlan(openai, query, history);
    const hits = await runStepsFn({
      plan: planObj.plan,
      // entities: planObj.entities,
      // expansions: planObj.expansions,
      embed: embedText,
      os: osClient,
      index: indexName
    });

    history.push({
      // intent: planObj.intent,
      // entities: planObj.entities,
      plan: planObj.plan,
      hit_count: hits.length
    });

    if (hits.length) {
      return hits;
    }
  }

  console.warn('[planAndExecute] max iterations reached with no coverage');
  return [];
}


module.exports = { buildPlan, planAndExecute };