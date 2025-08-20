from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any, Tuple
import json

load_dotenv()
client = OpenAI()

def split_transcript_into_lines(transcript: str) -> dict:
    """
    Splits a raw transcript string into a list of lines.
    Returns a dictionary, with line number to line mappings
    """
    split_transcript = transcript.split('.')
    line_mappings = {index: line.strip() for index, line in enumerate(split_transcript)}
    return line_mappings

def create_chunking_message(template_dict: dict, split_transcript: dict):
    if template_dict["length"] == "short":
        num_chunks = "2-4"
    elif template_dict["length"] == "medium":
        num_chunks = "5-7"
    else:
        num_chunks = "8-10"

    messages = [
        {
            "role": "system", 
            "content": f"Break the following transcript into {num_chunks} sections using the provided line numbers."
        },
        {
            "role": "user", 
            "content": str(split_transcript)
        }
    ]

    return messages

def annotate_transcript(split_transcript: dict, messages: list[dict]) -> list[dict]:
    """Create an annotated transcript, broken up into sections by line numbers"""

    class Section(BaseModel):
        """Schema for a transcript section"""
        header: str = Field(..., description="Header/title of the section")
        summary: str = Field(..., description = "Brief summary of the section")
        start_line: int = Field(..., description="Start line number of the section (inclusive)")
        end_line: int = Field(..., description="End line number of the section (inclusive)")

    class AnnotatedTranscript(BaseModel):
        """Schema for a transcript"""
        summary: str = Field(..., description = "Detailed summary of the transcript")
        sections: List[Section] = Field(..., description="List of structured transcript sections")
    
    res = client.beta.chat.completions.parse(
        model = "gpt-5",
        messages = messages,
        response_format = AnnotatedTranscript
    )

    annotated_transcript = res.choices[0].message.parsed.model_dump()
    messages.append({"role": "user", "content": str(annotated_transcript)})
    transcript_sections = annotated_transcript["sections"]

    return transcript_sections

def generate_section(prompt: str, transcript: str, source: str = "root", normalized_segment: Optional[dict] = None):
  """Generate a note section"""
  if source == "parent_chunk" and normalized_segment:
    content = f"header: {normalized_segment['header']} \n summary: {normalized_segment['summary']} \n transcript: {normalized_segment['transcript']}"
  else:
    content = transcript
    
  res = client.beta.chat.completions.create(
    model = "gpt-5",
    messages = [
      {
        "role": "system",
        "content": prompt
      },
      {
        "role": "user",
        "content": content
      }
    ]
  )
  section = res.choices[0].message.content
  return section

def normalize_segment(section: dict, split_transcript: dict) -> dict:
    """Normalize a transcript section by combining lines into a single string"""
    transcript = ""
    start_line = section["start_line"]
    end_line = section["end_line"]

    for i in range(start_line, end_line + 1):
        transcript += split_transcript[i] + ".\n"

    return {
        "header": section["header"], 
        "summary": section["summary"], 
        "transcript": transcript
    }

def visit_descendants(
    cur_node: Dict[str, Any],
    transcript: str,
    normalized_segment: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return (note_text, section_record) without mutating globals."""
    prompt = cur_node.get("prompt", "")
    source = cur_node.get("source", "root")

    output = ""
    if prompt:
        output = (generate_section(prompt, transcript, source, normalized_segment) or "").rstrip()

    record = {
        "id": cur_node.get("id"),
        "name": cur_node.get("name", ""),
        "prompt": prompt,
        "output": output,
        "children": []
    }

    parts: List[str] = [output] if output else []

    for child in cur_node.get("children", []) or []:
        child_note, child_record = visit_descendants(child, transcript, normalized_segment)
        if child_note:
            parts.append(child_note)
        record["children"].append(child_record)

    note = ("\n".join(parts).rstrip() + ("\n" if parts else ""))
    return note, record

def build_notes_and_flat_output(template_sections, transcript_segments, split_transcript, transcript):
    """Construct a note from its template, and return a flattened version of the note"""
    structured_output = {"transcript_sections": [], "chunks": []}
    notes = []
    checks = []

    def flatten_transcript_sections(rec):
        structured_output["transcript_sections"].append({
            "name": rec.get("name",""),
            "prompt": rec.get("prompt",""),
            "output": rec.get("output","")
        })
        for c in rec.get("children", []):
            flatten_transcript_sections(c)

    for section in template_sections:
        repeat_type = section.get("repeat", {}).get("type", "none")
        check = bool(section.get("check_hallucination", False))

        if repeat_type == "per_chunk":
            for seg in transcript_segments:
                checks.append(check)
                norm = normalize_segment(seg, split_transcript)
                chunk_entry = {"header": norm.get("header",""), "sections": []}
                structured_output["chunks"].append(chunk_entry)

                note, rec = visit_descendants(section, transcript, norm)

                chunk_entry["sections"].append(rec)
                notes.append(note)
        else:
            checks.append(check)
            note, rec = visit_descendants(section, transcript, normalized_segment=None)

            flatten_transcript_sections(rec)
            notes.append(note)

    return notes, checks, structured_output

def run(json_path: str, transcript_path: str):
    """Runs generate note workflow"""
    # parse template json
    with open(json_path, "r") as f:
        template = json.load(f)

    # annotate and split transcript
    with open(transcript_path, "r", encoding = "latin-1") as f:
        transcript = f.read()

    split_transcript = split_transcript_into_lines(transcript)
    chunk_message = create_chunking_message(template, split_transcript)
    transcript_sections = annotate_transcript(split_transcript, chunk_message)
    template_sections = template["sections"]

    # create note sections
    notes, checks, structured_output = build_notes_and_flat_output(
        template_sections,
        transcript_sections,
        split_transcript,
        transcript
    )

    return notes, checks, structured_output, transcript_sections, transcript