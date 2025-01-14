# https://docs.oracle.com/en-us/iaas/api/#/en/speech/20220101/TranscriptionJob/CreateTranscriptionJob

import oci

# Create a default config using DEFAULT profile in default location
# Refer to
# https://docs.cloud.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm#SDK_and_CLI_Configuration_File
# for more info
config = oci.config.from_file()


# Initialize service client with default config file
ai_speech_client = oci.ai_speech.AIServiceSpeechClient(config)


# Send the request to service, some parameters are not required, see API
# doc for more info
create_transcription_job_response = ai_speech_client.create_transcription_job(
    create_transcription_job_details=oci.ai_speech.models.CreateTranscriptionJobDetails(
        compartment_id="ocid1.test.oc1..<unique_ID>EXAMPLE-compartmentId-Value",
        input_location=oci.ai_speech.models.ObjectListFileInputLocation(
            location_type="OBJECT_LIST_FILE_INPUT_LOCATION",
            object_location=oci.ai_speech.models.ObjectLocation(
                namespace_name="EXAMPLE-namespaceName-Value",
                bucket_name="EXAMPLE-bucketName-Value",
                object_names=["EXAMPLE--Value"])),
        output_location=oci.ai_speech.models.OutputLocation(
            namespace_name="EXAMPLE-namespaceName-Value",
            bucket_name="EXAMPLE-bucketName-Value",
            prefix="EXAMPLE-prefix-Value"),
        display_name="EXAMPLE-displayName-Value",
        description="EXAMPLE-description-Value",
        additional_transcription_formats=["SRT"],
        model_details=oci.ai_speech.models.TranscriptionModelDetails(
            model_type="EXAMPLE-modelType-Value",
            domain="GENERIC",
            language_code="kn",
            transcription_settings=oci.ai_speech.models.TranscriptionSettings(
                diarization=oci.ai_speech.models.Diarization(
                    is_diarization_enabled=True,
                    number_of_speakers=12))),
        normalization=oci.ai_speech.models.TranscriptionNormalization(
            is_punctuation_enabled=True,
            filters=[
                oci.ai_speech.models.ProfanityTranscriptionFilter(
                    type="PROFANITY",
                    mode="TAG")]),
        freeform_tags={
            'EXAMPLE_KEY_reCDt': 'EXAMPLE_VALUE_XNxyFAM9Mof5OQ9ukRcz'},
        defined_tags={
            'EXAMPLE_KEY_2qVDu': {
                'EXAMPLE_KEY_zubEA': 'EXAMPLE--Value'}}),
    opc_retry_token="EXAMPLE-opcRetryToken-Value",
    opc_request_id="VL9WEIEFIUXSVY1WP1TJ<unique_ID>")

# Get the data from response
print(create_transcription_job_response.data)
