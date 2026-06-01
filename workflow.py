from langgraph.graph import START , END , StateGraph
from asyncio import graph
from asyncio import graph
from models import *
from agents import *
from langsmith import traceable

def entry_router (state:State) -> str :
    image = state.get('image_bytes')
    query = state.get('query')

    if not image :
        return 'query_rewritter'
    elif not query:
        return 'image'
    else :
        return ['query_rewritter','image']

def web_descion(state:State) -> str :
    web_stat = state.get('is_web')
    return 'use_web' if web_stat else 'use_ret'

def ranker (state:State) -> str :
    rank = state.get('rank')
    return 'accepted' if rank > 6 else 'rejected'

def is_cached (state:State) -> str :
    return 'jump' if state.get('cached') else 'continue'



class workflow():
    def __init__(self):
        self.responser = responser_agent
        self.query_filter = query_pii_agent
        self.image_filter = image_pii_agent
        self.merger = merging_agent
        self.is_cache = check_cache_agent
        self.k_web_getter = k_getter_use_web
        self.retriver = hyb_retriver_agent
        self.rewritter = rewrite_agent
        self.image = image_exp_agent
        self.web_searcher = web_scrapper_agent
        self.cacheing_agent = caching_agent
        self.ranker = ranker_agent
        self.entry = entry_router(state=State)

    def compile (self):
        graph = StateGraph(State)

        graph.add_node('entry_state',self.entry)
        graph.add_node("cache_check", self.is_cache)
        graph.add_node('responser',self.responser)
        graph.add_node('query_filter',self.query_filter)
        graph.add_node('image_filter',self.image_filter)
        graph.add_node('merger',self.merger)
        graph.add_node('k_web_getter',self.k_web_getter)    
        graph.add_node('retriver',self.retriver)
        graph.add_node('query_rewritter',self.rewritter)
        graph.add_node('image',self.image)
        graph.add_node('web_searcher',self.web_searcher)
        graph.add_node('ranker',self.ranker)
        graph.add_node('caching',self.caching_agent)

        graph.add_conditional_edges(START,'entry_state',{
            'query_rewritter':'query_filter',
            'image': 'image_filter'
        })

        graph.add_edge('query_filter','query_rewritter')
        graph.add_edge('image_filter','image')

        graph.add_edge('query_rewritter','merger')
        graph.add_edge('image','merger')

        graph.add_edge("merger", "cache_check")

        graph.add_conditional_edges("cache_check", is_cached, {
            "jump": END,
            "continue": "k_web_getter"
        })


        graph.add_conditional_edges('k_web_getter',web_descion,{
            'use_web' : 'web_searcher',
            'use_ret' : 'retriver'
        })

        graph.add_edge('web_searcher','responser')
        graph.add_edge('retriver','responser')

        graph.add_edge('responser','ranker')

        graph.add_conditional_edges('ranker',ranker,{
            'accepted': 'caching' ,
            'rejected': END
        })

        graph.add_edge('caching',END)
        
        return graph.compile()
    
    @traceable
    def run(self,intial_state=State):
        graph = self.compile()
        result = graph.invoke(intial_state)
        return result

