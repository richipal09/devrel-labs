# OCI CSV/JSON Translation Tool

## Introduction

There have been significant improvements in the world of language translation with the advancements of AI. With the advancement of Large Language Models (LLMs), we can now achieve language translation much easier than we did before.

In this solution, we will use OCI Language to enable the selective translation of specific columns in CSV files or keys in CSV or JSON documents, while preserving the original structure of the file. This use case is particularly useful for localizing data files while maintaining their format and untranslated fields.

The following OCI Services are present in this solution:
- **OCI Language** for document/field translation
- **OCI Compute** or **OCI Cloud Shell** for easily running the code present in this solution's repository, having authenticated this machine with the OCI SDK.

## 0. Prerequisites and setup

### Prerequisites

- Python 3.8 or higher
- OCI Account with Language Translation service enabled
- Required IAM Policies and Permissions
- Object Storage bucket for input/output files
- OCI CLI configured with proper credentials

### Docs

- [ISO Language Codes](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes)
- [OCI SDK](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm)

### Setup

1. Create an OCI account if you don't have one
2. Enable the Language Translation service in your tenancy if you haven't already
3. Set up OCI CLI and create API keys:

   ```bash
   # Install OCI CLI
   bash -c "$(curl -L https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh)"
   
   # Configure OCI CLI (this will create ~/.oci/config)
   oci setup config
   ```

4. Set up the appropriate IAM policies to use the service if you haven't ([see this link for more information](https://docs.oracle.com/en-us/iaas/language/using/policies.htm))
5. Create a bucket in Object Storage, where we'll put the file
6. Note your Object Storage namespace (visible in the OCI Console under Object Storage)

## Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/oracle-devrel/devrel-labs.git
   cd oci-csv-json-translation
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Update `config.yaml` with your settings:
   ```yaml
   # Language Translation Service Configuration
   language_translation:
     compartment_id: "ocid1.compartment.oc1..your-compartment-id"
     source_language: "en"  # ISO language code
     target_language: "es"  # ISO language code

   # Object Storage Configuration
   object_storage:
     namespace: "your-namespace"  # Your tenancy's Object Storage namespace
     bucket_name: "your-bucket-name"  # Bucket for CSV/JSON translations
   ```

4. Run the translation:
   ```bash
   # For CSV files (column numbers start from 1)
   python csv_json_translation.py csv input.csv output.csv 1 2 3

   # For JSON files
   python csv_json_translation.py json input.json output.json key1 key2
   ```

## Usage Examples

### CSV Translation
```bash
# Translate columns 1, 3, and 5 from English to Spanish
python csv_json_translation.py csv products.csv products_es.csv 1 3 5
```

### JSON Translation
```bash
# Translate 'name' and 'details' fields in a JSON file
python csv_json_translation.py json catalog.json catalog_es.json name details
```

## Configuration

The project uses three types of configuration:

1. **OCI Configuration** (`~/.oci/config`):
   - Created automatically by `oci setup config`
   - Contains your OCI authentication details
   - Used for API authentication

2. **Translation Configuration** (`config.yaml`):
   ```yaml
   # Language Translation Service Configuration
   language_translation:
     compartment_id: "ocid1.compartment.oc1..your-compartment-id"
     source_language: "en"
     target_language: "es"

   # Object Storage Configuration
   object_storage:
     namespace: "your-namespace"
     bucket_name: "your-bucket-name"
   ```

3. **Environment Variables** (optional, override config.yaml):
   - `OCI_COMPARTMENT_ID`: Your OCI compartment OCID
   - `OCI_SOURCE_LANG`: Source language code
   - `OCI_TARGET_LANG`: Target language code

### Configuration Priority

The configuration values are loaded in the following priority order:
1. Environment variables (if set)
2. Values from config.yaml
3. Default values (for language codes only: en -> es)

## Supported Languages

The service supports a wide range of languages. Common language codes include:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Chinese Simplified (zh-CN)
- Japanese (ja)

For a complete list of supported languages, refer to the OCI Documentation.

## Error Handling

The tool includes comprehensive error handling:
- Configuration validation
- Service availability checks
- File format validation
- Translation status monitoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 