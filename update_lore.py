import asyncio
import logging
from file_store import get_vector_store_id, upload_korvath_lore_file_to_vector_store

logger = logging.getLogger(__name__)

async def main():
    vector_id = await get_vector_store_id()
    result_str = await upload_korvath_lore_file_to_vector_store(vector_id)
    logger.info(result_str)

if __name__ == '__main__':
    asyncio.run(main())