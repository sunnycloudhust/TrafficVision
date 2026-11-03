from .cloud_storage import CloudObjectStorageSource


class HuggingFaceStreamingSource(CloudObjectStorageSource):
    """Use hf://datasets/<repo_id>/<path>. HF_TOKEN is read by the HF filesystem.

    No token is stored in configs. Shares the same sequential shard reader.
    """
