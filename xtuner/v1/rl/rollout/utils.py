import asyncio

import httpx

from xtuner.v1.utils import get_logger


logger = get_logger()


async def send_abort_request(client: httpx.AsyncClient, url: str, timeout: int = 60) -> tuple[str, bool]:
    worker_url = f"{url}/abort_request"
    try:
        response = await client.post(worker_url, json={"abort_all": True}, timeout=timeout)
        response.raise_for_status()
        logger.debug(f"Successfully sent abort request to {url}")
        return url, True
    except Exception as e:
        logger.error(f"Failed to send abort request to {url}: {e}")
        return url, False


async def pause_generation(rollout_ctl, pause_time_out=60.0):
    rollout_ctl_metadata = await rollout_ctl.get_rollout_metadata.remote()
    infer_server_url = list(rollout_ctl_metadata["server_url_dict"].values())
    async with httpx.AsyncClient() as client:
        tasks = [send_abort_request(client, url, timeout=pause_time_out) for url in infer_server_url]
        results = await asyncio.gather(*tasks)

    failed_workers = [url for url, success in results if not success]
    succeeded_count = len(infer_server_url) - len(failed_workers)

    if failed_workers:
        logger.warning(
            f"Abort requests completed. Succeeded: {succeeded_count}, "
            f"Failed: {len(failed_workers)}. Failed workers: {failed_workers}"
        )
    else:
        logger.info(f"All {succeeded_count} abort requests sent successfully.")


async def continue_generation(rollout_ctl):
    return await rollout_ctl.continue_generation.remote()
