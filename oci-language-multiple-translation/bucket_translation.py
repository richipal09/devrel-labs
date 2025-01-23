#!/usr/bin/env python3

import oci
import yaml
import sys
import time
import datetime
from pathlib import Path

def load_config():
    """Load configuration from config.yaml file"""
    try:
        with open("config.yaml", "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config.yaml: {str(e)}")
        sys.exit(1)

def init_clients():
    """Initialize OCI clients"""
    try:
        # Initialize the AI Language client using default OCI config
        config = oci.config.from_file()
        ai_client = oci.ai_language.AIServiceLanguageClient(config=config)
        object_storage = oci.object_storage.ObjectStorageClient(config=config)
        return ai_client, object_storage
    except Exception as e:
        print(f"Error initializing OCI clients: {str(e)}")
        sys.exit(1)

def generate_job_name():
    """Generate a unique job name with timestamp"""
    current_date = datetime.date.today()
    current_time = datetime.datetime.now()
    return f"translation_job_{current_date.strftime('%Y-%m-%d')}T{current_time.strftime('%H-%M-%S')}"

def translate_documents(ai_client, object_storage, config):
    """Translate all documents in the source bucket"""
    try:
        # Get configuration values
        compartment_id = config["language_translation"]["compartment_id"]
        source_bucket = config["language_translation"]["source_bucket"]
        target_bucket = config["language_translation"]["target_bucket"]
        source_language = config["language_translation"]["source_language"]
        target_language = config["language_translation"]["target_language"]
        
        # Get namespace
        namespace = object_storage.get_namespace().data
        print(f"Using namespace: {namespace}")
        
        # Create input location for all files in the bucket
        input_location = oci.ai_language.models.ObjectStoragePrefixLocation(
            namespace_name=namespace,
            bucket_name=source_bucket
        )

        # Set up model metadata for translation
        model_metadata_details = oci.ai_language.models.ModelMetadataDetails(
            model_type="PRE_TRAINED_TRANSLATION",
            language_code=source_language,
            configuration={
                "targetLanguageCodes": oci.ai_language.models.ConfigurationDetails(
                    configuration_map={"languageCodes": target_language}
                )
            }
        )

        # Set up output location
        output_location = oci.ai_language.models.ObjectPrefixOutputLocation(
            namespace_name=namespace,
            bucket_name=target_bucket
        )

        # Create job details
        create_job_details = oci.ai_language.models.CreateJobDetails(
            display_name=generate_job_name(),
            compartment_id=compartment_id,
            input_location=input_location,
            model_metadata_details=[model_metadata_details],
            output_location=output_location
        )
        
        # Create and submit translation job
        print("Creating translation job...")
        job_response = ai_client.create_job(
            create_job_details=create_job_details
        )
        
        job_id = job_response.data.id
        print(f"Translation job created with ID: {job_id}")
        
        # Monitor job status
        while True:
            job_status = ai_client.get_job(job_id=job_id)
            current_state = job_status.data.lifecycle_state
            print(f"{datetime.datetime.now()}: Job status: {current_state}")
            
            if current_state in ["SUCCEEDED", "FAILED"]:
                break
            time.sleep(30)
        
        if current_state == "SUCCEEDED":
            print(f"Translation completed successfully! Check the {target_bucket} bucket for results")
            return True
        else:
            print(f"Translation failed with status: {current_state}")
            return False
        
    except Exception as e:
        print(f"Error during translation: {str(e)}")
        return False

def main():
    try:
        # Load configuration
        config = load_config()
        
        # Initialize clients
        ai_client, object_storage = init_clients()
        
        # Start translation
        success = translate_documents(ai_client, object_storage, config)
        
        if success:
            print("Translation completed successfully!")
        else:
            print("Translation failed.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
