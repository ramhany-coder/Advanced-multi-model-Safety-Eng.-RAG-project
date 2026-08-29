from gptcache import cache
from gptcache.adapter.api import get as cache_get, put as cache_put
from gptcache.processor.pre import get_prompt

cache.init(pre_embedding_func=get_prompt)


def check_cache_agent(state) -> dict:
    query = state.get('merged')
    result = cache_get(query)
    if result:
        return {'cached': True, "response": result}
    else:
        return {"cached": False}


def caching_agent(state) -> dict:
    caching_stat = state.get('cached')
    if not caching_stat:
        query = state.get('merged')
        response = state.get('response')
        if response and query:
            cache_put(query, response)
