"""Download one direct HTTP/cloud object to a verified runtime cache.

Google Drive confirmation/share pages are NOT archive URLs. Use gdown for those,
then pass downloaded images/XML to prepare_ua_detrac.py.
"""

import argparse
from trafficvision.data.cache import cached_file

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--cache-dir", default="/tmp/trafficvision-downloads")
    p.add_argument("--sha256")
    p.add_argument("--size", type=int)
    args = p.parse_args()
    print(cached_file(args.url, args.cache_dir, args.sha256, args.size))
