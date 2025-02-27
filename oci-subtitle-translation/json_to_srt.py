import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import re

def parse_timestamp(time_str: str) -> str:
    """Convert timestamp from 's' format to 'HH:MM:SS,mmm' format"""
    # Remove 's' suffix and convert to float
    seconds = float(time_str.rstrip('s'))
    
    # Calculate hours, minutes, seconds and milliseconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    
    # Format timestamp
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def get_sentences(transcription: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split transcription into sentences based on punctuation"""
    sentences = []
    current_sentence = []
    
    for token in transcription:
        # Skip empty or invalid tokens
        if not isinstance(token, dict) or "type" not in token:
            continue
            
        current_sentence.append(token)
        
        # If we find punctuation, end the current sentence
        if token["type"] == "PUNCTUATION":
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
    
    # Add any remaining tokens as the last sentence
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences

def create_srt_entry(index: int, sentence_tokens: List[Dict[str, Any]]) -> str:
    """Create a single SRT entry from a sentence"""
    # Get start time from first word after last punctuation
    start_time = None
    for i, token in enumerate(sentence_tokens):
        if token["type"] == "PUNCTUATION":
            if i + 1 < len(sentence_tokens):
                start_time = sentence_tokens[i + 1]["startTime"]
            break
    
    # If no punctuation found or it's the last token, use first token's start time
    if not start_time:
        start_time = sentence_tokens[0]["startTime"]
    
    # Get end time from the last token
    end_time = sentence_tokens[-1]["endTime"]
    
    # Convert timestamps
    start_timestamp = parse_timestamp(start_time)
    end_timestamp = parse_timestamp(end_time)
    
    # Build sentence text (join all words, including punctuation)
    text = ""
    for token in sentence_tokens:
        if token["type"] == "WORD":
            text += " " + token["token"]
        elif token["type"] == "PUNCTUATION":
            text = text.rstrip() + token["token"]
    text = text.strip()
    
    # Format SRT entry
    return f"{index}\n{start_timestamp} --> {end_timestamp}\n{text}\n"

def convert_json_to_srt(input_file: str, output_file: str):
    """Convert JSON transcription file to SRT format"""
    try:
        # Read JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract transcriptions - handle both array and direct object formats
        transcriptions = data.get("transcriptions", [])
        if transcriptions and isinstance(transcriptions, list):
            tokens = transcriptions[0].get("tokens", [])
        else:
            # Try direct token access if not in transcriptions array
            tokens = data.get("tokens", [])
        
        if not tokens:
            raise ValueError("No tokens found in JSON file")
        
        # Split into sentences
        sentences = get_sentences(tokens)
        
        # Create SRT content
        srt_content = ""
        for i, sentence in enumerate(sentences, 1):
            srt_content += create_srt_entry(i, sentence) + "\n"
        
        # Write SRT file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(srt_content.strip())
        
        print(f"✓ Successfully converted {input_file} to {output_file}")
        print(f"✓ Created {len(sentences)} subtitle entries")
        
    except Exception as e:
        print(f"✗ Error converting file: {str(e)}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Convert JSON transcription to SRT format")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output SRT file")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert file
    convert_json_to_srt(args.input, args.output)

if __name__ == "__main__":
    main() 