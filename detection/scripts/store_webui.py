from google.cloud import storage

bucket_name = "processed_webui"
source_file_name = "downloads/balanced_7k_processed.zip"
destination_blob_name = "balanced_7k_processed.zip"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(source_file_name)
print(f"File {source_file_name} uploaded to {destination_blob_name}.")
