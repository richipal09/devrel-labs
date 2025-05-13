import oci
import yaml
import argparse
import sys
import os
import re
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
    config = oci.config.from_file(profile_name="DEVRELCOMM")
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

def parse_srt(srt_text):
    srt_entries = []
    
    for block in srt_text.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        index, timestamp, *text_lines = lines  
        text = " ".join(text_lines)  
        srt_entries.append((index, timestamp, text))

    return srt_entries
srt_entries = parse_srt(srt_content)
log_step(f"Extracted subtitles from SRT file.")

translated_entries = []
for index, timestamp, text in srt_entries:
    try:
        translate_request = oci.ai_language.models.TranslateTextDetails(
            text=text,
            source_language_code="auto",
            target_language_code=args.target_language
        )
        translation_response = ai_translation_client.translate_text(translate_request)
        translated_text = translation_response.data.translated_text if translation_response and translation_response.data else text
        log_step(f"Translated: {text} --> {translated_text}")
    except Exception as e:
        log_step(f"Translation failed for: {text}. Error: {str(e)}", True)
        translated_text = text 

    translated_entries.append((index, timestamp, translated_text)) 


log_step(f"Translation completed successfully with {len(translated_entries)} entries.")

def rebuild_srt(translated_entries):
    srt_output = []
    for index, timestamp, translated_text in translated_entries:
        srt_output.append(f"{index}\n{timestamp}\n{translated_text}\n")
    
    return "\n".join(srt_output)

translated_srt_content = rebuild_srt(translated_entries)

try:
    object_storage_client.put_object(namespace, bucket_name, output_file, translated_srt_content.encode('utf-8'))
    log_step(f"Translated SRT uploaded to OCI as {output_file}")
except Exception as e:
    log_step(f"Failed to upload translated SRT to OCI: {str(e)}", True)
    sys.exit(1)
