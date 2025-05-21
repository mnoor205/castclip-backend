import boto3
import os

s3 = boto3.client(
    "s3",
    aws_access_key_id="ecde4491fd6fcfa4606da4e30a2ae7ce",
    aws_secret_access_key="fb055a76cd7dfe950683a443bfef15b29a3d03a134f5068f70315baa48a4b9af",
    endpoint_url="https://27478de5040614b0328a6976a52b57d1.r2.cloudflarestorage.com/",
    region_name="auto",  # Required even if ignored
    config=boto3.session.Config(signature_version='s3v4')  # VERY IMPORTANT
)

# Try to list the objects
response = s3.list_objects_v2(Bucket="ai-podcast-clipper", Prefix="test1/")
for obj in response.get("Contents", []):
    print("Found:", obj["Key"])