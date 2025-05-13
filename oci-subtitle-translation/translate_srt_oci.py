import oci
import yaml
import argparse
import sys
import os
from datetime import datetime

def log_step(message, is_error=False):
    """Prints a formatted log message with a timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = "ERROR" if is_error else "INFO"
    print(f"[{timestamp}] {prefix}: {message}")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Translate an SRT file using OCI AI Translation')
parser.add_argument('--input-file', required=True, help='Input SRT file name in the configured bucket')
parser.add_argument('--target-language', required=True, help='Target language code (e.g fr, es, de)')
args = parser.parse_args()

# Generate output filename
input_filename = os.path.splitext(args.input_file)[0]  # Remove extension
output_file = f"{input_filename}_{args.target_language}.srt"

log_step(f"Starting translation of {args.input_file} to {args.target_language}")

# Create a default config using DEFAULT profile in default location
try:
    config = oci.config.from_file(profile_name="comm")
    log_step("Successfully loaded OCI configuration")
except Exception as e:
    log_step(f"Failed to load OCI configuration: {str(e)}", True)
    sys.exit(1)

# Initialize service client with default config file
try:
    ai_language_client = oci.ai_language.AIServiceLanguageClient(config)
    log_step("Successfully initialized AI Language client")
except Exception as e:
    log_step(f"Failed to initialize AI Language client: {str(e)}", True)
    sys.exit(1)

# Load bucket details from config.yaml
def load_config():
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            log_step("Successfully loaded config.yaml")
            return config
    except Exception as e:
        log_step(f"Failed to load config.yaml: {str(e)}", True)
        sys.exit(1)

config_yaml = load_config()

object_storage_client = oci.object_storage.ObjectStorageClient(config)

# Read SRT file from OCI bucket
try:
    namespace = config_yaml['speech']['namespace']
    bucket_name = config_yaml['speech']['bucket_name']
    object_name = args.input_file

    get_object_response = object_storage_client.get_object(namespace, bucket_name, object_name)
    srt_content = get_object_response.data.text.strip()
    log_step(f"Loaded SRT file from OCI: {args.input_file}")

except Exception as e:
    log_step(f"Failed to read SRT file from OCI Object Storage: {str(e)}", True)
    sys.exit(1)

try:
    # Check text length before translation
    text_length = len(srt_content)
    log_step(f"Text length: {text_length} characters")
    
    if text_length > 5000:
        log_step("Text length exceeds 5000 characters limit. Translation cannot proceed.", True)
        sys.exit(1)
    
    # Split SRT content into chunks of max 5000 characters
    def split_text_into_chunks(text, max_chunk_size=5000):
        chunks = []
        current_position = 0
        text_length = len(text)
        
        while current_position < text_length:
            chunk_end = min(current_position + max_chunk_size, text_length)
            # Try to find a newline to make cleaner splits
            if chunk_end < text_length and text[chunk_end] != '\n':
                # Look for the last newline in this chunk
                last_newline = text.rfind('\n', current_position, chunk_end)
                if last_newline > current_position:
                    chunk_end = last_newline + 1
            
            chunks.append(text[current_position:chunk_end])
            current_position = chunk_end
        
        return chunks
    
    # Split content into manageable chunks
    srt_chunks = split_text_into_chunks(srt_content)
    log_step(f"Split SRT content into {len(srt_chunks)} chunks for translation")
    
    # Create batch translation request with multiple documents
    documents = []
    for i, chunk in enumerate(srt_chunks):
        documents.append({
            "key": str(i+1),
            "text": chunk,
            "languageCode": "auto"
        })
    
    batch_translation_details = oci.ai_language.models.BatchLanguageTranslationDetails(
        documents=documents,
        target_language_code=args.target_language
    )
    
    # Execute batch translation
    translation_response = ai_language_client.batch_language_translation(batch_translation_details)
    
    # Get the translated content
    if translation_response.data and translation_response.data.documents:
        translated_content = translation_response.data.documents[0].translated_text
        log_step("Successfully translated the entire SRT file")
    else:
        log_step("No translation result received", True)
        sys.exit(1)

except Exception as e:
    log_step(f"Batch translation failed: {str(e)}", True)
    sys.exit(1)

# Upload the translated content back to OCI
try:
    object_storage_client.put_object(namespace, bucket_name, output_file, translated_content.encode('utf-8'))
    log_step(f"Translated SRT uploaded to OCI as {output_file}")
except Exception as e:
    log_step(f"Failed to upload translated SRT to OCI: {str(e)}", True)
    sys.exit(1) 