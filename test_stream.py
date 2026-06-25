import asyncio, os
from dotenv import load_dotenv
load_dotenv()
from langchain_redis import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI

async def main():
    cache = RedisSemanticCache(redis_url=os.getenv('REDIS_URL'), embeddings=OpenAIEmbeddings(), distance_threshold=0.15)
    set_llm_cache(cache)
    llm = ChatOpenAI(model='gpt-4o-mini')
    print('Invoking stream...')
    async for chunk in llm.astream('I wanna know about return policy'):
        print(chunk.content, end='', flush=True)
    print()

asyncio.run(main())
