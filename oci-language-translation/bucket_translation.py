#!/usr/bin/env python3

import oci
import yaml
import sys
import time
from pathlib import Path

def load_config():
    """Load configuration from config.yaml file"""
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

def init_clients(config):
    """Initialize OCI clients"""
    # Initialize the AI Language client
    ai_client = oci.ai_language.AIServiceLanguageClient(
        oci.config.from_file(
            file_location="config.yaml",
            profile_name="DEFAULT"
        )
    )
    
    # Initialize Object Storage client
    object_storage = oci.object_storage.ObjectStorageClient(
        oci.config.from_file(
            file_location="config.yaml",
            profile_name="DEFAULT"
        )
    )
    
    return ai_client, object_storage

def list_objects_in_bucket(object_storage, namespace, bucket_name):
    """List all objects in a bucket"""
    list_objects_response = object_storage.list_objects(
        namespace_name=namespace,
        bucket_name=bucket_name
    )
    return [obj.name for obj in list_objects_response.data.objects]

def translate_documents(ai_client, config):
    """Translate all documents in the source bucket"""
    try:
        # Get configuration values
        compartment_id = config["language_translation"]["compartment_id"]
        source_bucket = config["language_translation"]["source_bucket"]
        target_bucket = config["language_translation"]["target_bucket"]
        source_language = config["language_translation"]["source_language"]
        target_language = config["language_translation"]["target_language"]
        
        # Create batch document translation job
        create_batch_job_response = ai_client.create_batch_document_translation_job(
            create_batch_document_translation_job_details=oci.ai_language.models.CreateBatchDocumentTranslationJobDetails(
                compartment_id=compartment_id,
                display_name=f"Batch_Translation_{time.strftime('%Y%m%d_%H%M%S')}",
                source_language_code=source_language,
                target_language_code=target_language,
                input_location=oci.ai_language.models.ObjectStorageLocation(
                    bucket_name=source_bucket,
                    namespace_name=namespace
                ),
                output_location=oci.ai_language.models.ObjectStorageLocation(
                    bucket_name=target_bucket,
                    namespace_name=namespace
                )
            )
        )
        
        job_id = create_batch_job_response.data.id
        print(f"Translation job created with ID: {job_id}")
        
        # Monitor job status
        while True:
            job_status = ai_client.get_batch_document_translation_job(
                batch_document_translation_job_id=job_id
            ).data.lifecycle_state
            
            print(f"Job status: {job_status}")
            if job_status in ["SUCCEEDED", "FAILED"]:
                break
            time.sleep(30)
        
        return job_status == "SUCCEEDED"
        
    except Exception as e:
        print(f"Error during translation: {str(e)}")
        return False

def main():
    try:
        # Load configuration
        config = load_config()
        
        # Initialize clients
        ai_client, object_storage = init_clients(config)
        
        # Start translation
        success = translate_documents(ai_client, config)
        
        if success:
            print("Translation completed successfully!")
        else:
            print("Translation failed.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
