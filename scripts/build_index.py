import asyncio

from app.container import AppContainer
from app.core.config import get_settings


async def main() -> None:
    settings = get_settings()
    container = AppContainer(settings)
    await container.initialize()
    container.kb_service.rebuild_from_sources()
    await container.vector_store.ensure_index(rebuild=True)
    print(
        f"Rebuilt index with {len(container.vector_store.vectors)} embeddings "
        f"for {len(container.kb_service.chunks)} chunks."
    )


if __name__ == "__main__":
    asyncio.run(main())
