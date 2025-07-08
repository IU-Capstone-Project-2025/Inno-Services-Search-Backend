import asyncio
import re
import time

import lancedb
from langdetect import detect
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from src.api.logging_ import logger
from src.config import settings
from src.ml_service.text import clean_text
from src.modules.sources_enum import ALL_SOURCES, InfoSources

model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)


def translate_to_russian(text: str) -> str:
    tokenizer.src_lang = "en"
    encoded = tokenizer(text, return_tensors="pt")
    generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("ru"))
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


async def search_maps(
    tbl,
    query_text: str,
    query_emb,
    limit: int,
    area_id: str | None = None,
) -> list[dict]:
    """
    1) Full-text search on map chunks
    2) Vector search on the same table
    3) If area_id given, prioritize those chunks
    4) Merge + dedupe by mongo_id
    """
    # 1) Full-text search
    try:
        fts_hits = await tbl.search_text(query_text).limit(limit).to_pylist()
    except Exception:
        fts_hits = []
    fts_results = [
        {
            "resource": InfoSources.maps.value,
            "mongo_id": hit["mongo_id"],
            "score": float(hit.get("_score", hit.get("_relevance_score", 0.0))),
            "content": clean_text(hit["content"]),
            "chunk_number": hit.get("chunk_number", 0),
        }
        for hit in fts_hits
    ]

    # 2) Vector search
    emb_df = (
        await tbl.query()
        .nearest_to(query_emb)
        .distance_type("cosine")
        .limit(settings.ml_service.bi_encoder_search_limit_per_table)
        .to_pandas()
    )
    emb_df["score"] = 1 - emb_df["_distance"]

    # 3) Если area_id указан — фильтруем по нему
    if area_id:
        mask = emb_df["content"].str.contains(f'area_id="{area_id}"')
        filtered = emb_df[mask]
        chosen = filtered if not filtered.empty else emb_df
    else:
        chosen = emb_df

    emb_selected = chosen.sort_values("score", ascending=False).head(limit)
    emb_results = [
        {
            "resource": InfoSources.maps.value,
            "mongo_id": row["mongo_id"],
            "score": float(row["score"]),
            "content": clean_text(row["content"]),
            "chunk_number": row.get("chunk_number", 0),
        }
        for _, row in emb_selected.iterrows()
    ]

    # 4) Merge + dedupe by mongo_id, keep highest score
    combined = fts_results + emb_results
    unique: dict[str, dict] = {}
    for r in combined:
        key = r["mongo_id"]
        if key not in unique or r["score"] > unique[key]["score"]:
            unique[key] = r

    return list(unique.values())


async def search_pipeline(
    query: str,
    resources: list[InfoSources] = ALL_SOURCES,
    limit: int = 5,
):
    start_time = time.perf_counter()
    logger.info(f"🔍 Searching for {query!r} in {resources}")

    original_query = query
    # detect language
    try:
        query_lang = detect(query)
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        query_lang = None

    # translate if needed
    if query_lang == "en" and InfoSources.residents in resources:
        logger.info("Translating English query for residents resource")
        search_query = translate_to_russian(query)
        logger.info(f"Translated query: {search_query!r}")
    else:
        search_query = query

    # prepare embedding
    bi_start = time.perf_counter()
    cleaned = clean_text(search_query)
    if settings.ml_service.infinity_url:
        import src.ml_service.infinity as svc
    else:
        import src.ml_service.non_infinity as svc
    query_emb = (await svc.embed([cleaned], task="query"))[0]
    bi_time = time.perf_counter() - bi_start
    logger.info(f"⏱️ Bi-encoder: {bi_time:.3f}s")

    lance_db = await lancedb.connect_async(settings.ml_service.lancedb_uri)
    all_results: list[dict] = []
    area_pattern = re.compile(r"(\d+(?:[.,]\d+)?)")
    area_match = area_pattern.search(query)
    area_id = area_match.group(1).replace(",", ".") if area_match else None

    # iterate over tables
    db_start = time.perf_counter()
    for res in resources:
        table_name = f"chunks_{res.value}"
        if table_name not in await lance_db.table_names():
            continue

        tbl = await lance_db.open_table(table_name)

        if res is InfoSources.maps:
            # для карт — комбинированный FTS+vector
            maps_hits = await search_maps(tbl, search_query, query_emb, limit, area_id)
            all_results.extend(maps_hits)
        else:
            # только векторный поиск
            df = (
                await tbl.query()
                .nearest_to(query_emb)
                .distance_type("cosine")
                .limit(settings.ml_service.bi_encoder_search_limit_per_table)
                .to_pandas()
            )
            for _, row in df.iterrows():
                score = row.get("_relevance_score") or row.get("_score") or (1 - row.get("_distance", 1))
                all_results.append(
                    {
                        "resource": res.value,
                        "mongo_id": row["mongo_id"],
                        "score": float(score),
                        "content": clean_text(row["content"]),
                        "chunk_number": row.get("chunk_number", 0),
                    }
                )

    db_time = time.perf_counter() - db_start
    logger.info(f"⏱️ DB queries: {db_time:.3f}s, items: {len(all_results)}")

    # rerank with cross-encoder
    cross_time = 0.0
    if all_results:
        logger.info("🔄 Cross-encoder reranking…")
        docs = [r["content"] for r in all_results]
        cross_start = time.perf_counter()
        rankings = await svc.rerank(clean_text(query), docs, top_n=limit)
        cross_time = time.perf_counter() - cross_start

        # rebuild list in ranked order
        reranked = []
        for rank in rankings:
            orig = all_results[rank.index]
            reranked.append({**orig, "score": float(rank.relevance_score)})
        all_results = reranked
    logger.info(f"⏱️ Cross-encoder: {cross_time:.3f}s")

    # dedupe across resources
    unique: dict[tuple[str, str], dict] = {}
    for r in all_results:
        key = (r["resource"], r["mongo_id"])
        if key not in unique or r["score"] > unique[key]["score"]:
            unique[key] = r
    final = sorted(unique.values(), key=lambda x: x["score"], reverse=True)

    total = time.perf_counter() - start_time
    logger.info(f"⏱️ Total pipeline: {total:.3f}s")

    return {
        "results": final,
        "original_query": original_query,
        "query_lang": query_lang,
        "query_lang_name": {"en": "English", "ru": "Russian"}.get(query_lang, "other"),
    }


if __name__ == "__main__":
    logger.info("📥 Starting search pipeline…")
    q = "How much does room for 2 people rent cost?"
    data = asyncio.run(search_pipeline(q, resources=ALL_SOURCES))
    for i, r in enumerate(data["results"], 1):
        logger.info(f"{i}. ({r['resource']}) [{r['score']:.3f}]: {r['content']}")
