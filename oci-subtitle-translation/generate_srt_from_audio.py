# https://docs.oracle.com/en-us/iaas/api/#/en/speech/20220101/TranscriptionJob/CreateTranscriptionJob

import oci

# Create a default config using DEFAULT profile in default location
# Refer to
# https://docs.cloud.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm#SDK_and_CLI_Configuration_File
# for more info
config = oci.config.from_file()


# Initialize service client with default config file
ai_speech_client = oci.ai_speech.AIServiceSpeechClient(config)


# Load config from yaml file
def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

config_yaml = load_config()

# Send the request to service, some parameters are not required, see API
# doc for more info
create_transcription_job_response = ai_speech_client.create_transcription_job(
    create_transcription_job_details=oci.ai_speech.models.CreateTranscriptionJobDetails(
        compartment_id=config_yaml['speech']['compartment_id'],
        input_location=oci.ai_speech.models.ObjectListFileInputLocation(
            location_type="OBJECT_LIST_FILE_INPUT_LOCATION", 
            object_location=oci.ai_speech.models.ObjectLocation(
                namespace_name=config_yaml['speech']['namespace'],
                bucket_name=config_yaml['speech']['bucket_name'],
                object_names=["FILE_NAMES"])),
        output_location=oci.ai_speech.models.OutputLocation(
            namespace_name=config_yaml['speech']['namespace'],
            bucket_name=config_yaml['speech']['bucket_name'],
            prefix="transcriptions"),
        display_name=f"Transcription_{args.input_file}",
        description=f"Transcription job for {args.input_file}",
        additional_transcription_formats=["SRT"],
        model_details=oci.ai_speech.models.TranscriptionModelDetails(
            domain="GENERIC",
            language_code="en",
            transcription_settings=oci.ai_speech.models.TranscriptionSettings(
                diarization=oci.ai_speech.models.Diarization(
                    is_diarization_enabled=True,
                    number_of_speakers=2))),
        normalization=oci.ai_speech.models.TranscriptionNormalization(
            is_punctuation_enabled=True,
            filters=[
                oci.ai_speech.models.ProfanityTranscriptionFilter(
                    type="PROFANITY",
                    mode="TAG")]),
        freeform_tags={},
        defined_tags={}))

# Get the data from response
print(create_transcription_job_response.data)
