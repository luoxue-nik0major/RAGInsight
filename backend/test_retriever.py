import asyncio
import sys
sys.path.insert(0, 'd:/普通下载/大创/接下来主要的任务/backend')

from app.services.retriever import VectorRetrieverAdapter

async def main():
    adapter = VectorRetrieverAdapter()
    result = await adapter.retrieve("李白 月亮", top_k=5)
    print(f"Total found: {result['total_found']}")
    for c in result['chunks']:
        print(f"  Score: {c['relevance_score']:.3f} | {c['content'][:80]}")

asyncio.run(main())
