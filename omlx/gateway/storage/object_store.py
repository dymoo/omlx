"""Opt-in bounded traces. Credentials come from the standard AWS chain."""

import asyncio
import gzip
import json
from datetime import UTC, datetime, timedelta


class ObjectStore:
    def __init__(self, bucket, endpoint_url=None):
        import boto3

        self.client = boto3.client("s3", endpoint_url=endpoint_url)
        self.bucket = bucket

    async def put(self, request_id, document, ttl_days):
        now = datetime.now(UTC)
        key = f"traces/{now:%Y/%m/%d}/{request_id}.json.gz"
        body = gzip.compress(json.dumps(document, default=str).encode())
        # Expires is metadata, NOT an S3 deletion policy. Operators must install
        # bucket lifecycle rules for each retention tag before enabling full logs.
        await asyncio.to_thread(
            self.client.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ContentEncoding="gzip",
            ServerSideEncryption="AES256",
            Tagging=f"retention_days={ttl_days}",
            Expires=now + timedelta(days=ttl_days),
        )
        return key
