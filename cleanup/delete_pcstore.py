from pinecone import Pinecone
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("polygist")

delete_before = (datetime.now(timezone.utc) - timedelta(days=1)).timestamp()

spaces = index.list_namespaces()
for page in spaces:
    for ns in page.namespaces:
        name = ns.name
        try:
            before = index.describe_index_stats().namespaces.get(name)
            count_before = before.vector_count if before else 0

            index.delete(
                filter={"created_at": {"$lte": delete_before}},
                namespace=name
                )

            after = index.describe_index_stats().namespaces.get(name)
            count_after = after.vector_count if after else 0

            if count_after < count_before:
                print(f"Namespace {name}: deleted {count_before - count_after} records")
            else:
                print(f"Namespace {name}: no matching records to delete")

        except Exception as e:
            print(f"Error on namespace {name}: {e}")