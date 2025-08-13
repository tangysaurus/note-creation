from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from typing import List

load_dotenv()
client = OpenAI()

class SentenceLine(BaseModel):
    """Schema for a transcript line"""
    line_number: int = Field(..., description="Line number in the transcript")
    text: str = Field(..., description="The sentence on this line")

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

class NoteSection(BaseModel):
    """Schema for a note section"""
    header: str = Field(..., description="Short, catchy header/title of the section formatted in markdown using '##'. Include a relevant emoji.")
    one_liner: str = Field(..., description = "Short, interesting, one-liner that encapsulates section. Put the one-liner in "". Use > 'quote' markdown")
    content: str = Field(..., description = "Detailed, bullet-points illustrating the section")

class Extras(BaseModel):
    """Schema for additional content"""
    headline: str = Field(..., description = "Short, catchy title that encapsulates this lecture. Formatted in markdown using '##'.")
    summary: str = Field(..., description = "Short, engaging outline of the lecture. Begin with 'This lecture covers...'. Italicize in markdown using '*'.")
    todo_list: str = Field(
                            ..., 
                            description = 
                                    """
                                        Short list of the most important todos and next steps mentioned by the professor in the lecture. 
                                        Prioritize readings, exams, and assignments. 
                                        Include deadlines. 
                                        Format the title 'Todo List' in markdown using '###'.
                                        Use markdown format to **bold** parts of each todo. Keep each todo to 15 words or less. 
                                        Include markdown checkboxes '- [ ]' before each todo.
                                    """
                        )
    glossary: str = Field(..., description = "Glossary of key terms and definitions used throughout the lecture. Format the title 'Glossary' in markdown using '###'. Format the glossary as a markdown table with a divider '---' at the end.")
    quiz: str = Field(..., description = "Written quiz that tests a student's understanding of the lecture. Format the title 'quiz' in markdown using '###'.")
    answers: str = Field(..., description = "Answers to the written quiz. Format the title 'answers' in markdown using '###'.")

class Eval(BaseModel):
    """Schema for eval"""
    hallucinations: list[str] = Field(..., description = "Explanations of errors and inconsistencies")
    revisions: list[str] = Field(..., description = "Revisions made to correct the corresponding hallucination")
    score: int = Field(..., description = "Score (from 0% to 100%) that represents the percentage of the section that contained hallucinations")

def split_transcript_into_lines(transcript: str) -> dict:
    """
    Splits a raw transcript string into a list of lines.
    Returns a dictionary, with line number to line mappings
    """
    split_transcript = transcript.split('.')
    line_mappings = {index: line.strip() for index, line in enumerate(split_transcript)}
    return line_mappings

def annotate_transcript(split_transcript: dict) -> list[dict]:
    """Create an annotated transcript, broken up into sections by line numbers"""
    res = client.beta.chat.completions.parse(
        model = "gpt-4.1",
        messages = [
            {
                "role": "system", 
                "content": "Break the following transcript into 5 or 6 sections using the provided line numbers."
            },
            {
                "role": "user", 
                "content": str(split_transcript)
            }
        ],
        response_format = AnnotatedTranscript
    )
    transcript_sections = res.choices[0].message.parsed.model_dump()["sections"]
    return transcript_sections

def normalize_section(section: dict, split_transcript: dict) -> dict:
    """Normalize a transcript section by combining lines into a single string"""
    transcript = ""
    start_line = section["start_line"]
    end_line = section["end_line"]

    for i in range(start_line, end_line):
        transcript += split_transcript[i] + "\n"

    return {
        "header": section["header"], 
        "summary": section["summary"], 
        "transcript": transcript
    }

def generate_note_section(normalized_section: dict) -> dict:
    """Combine transcript section metadata and content to create a note section"""
    res = client.beta.chat.completions.parse(
        model = "gpt-4.1",
        messages = [
            {
                "role": "system", 
                "content": 
                    """
                    You are a course creator for a college class. 
                    Use the following section header, summary, and transcript to create a detailed note.
                    Indicate subheaders using '##', bold key terms and concepts for emphasis, use bullet points to represent lists, and indentation to indicate dependencies.
                    Format headers, one-liners, and text in markdown. Format math equations and symbols in katex.
                    """
            },
            {
                "role": "user", 
                "content": f"header: {normalized_section['header']} \n summary: {normalized_section['summary']} \n transcript: {normalized_section['transcript']}"
            }
        ],
        response_format = NoteSection
    )
    note_section = res.choices[0].message.parsed.model_dump()
    return note_section

def compile_note_section(note_section: dict) -> str:
    """Combine note section components into a string"""
    combined = note_section["header"] + "\n" + note_section["one_liner"] + "\n" + note_section["content"]
    return combined

def generate_extras(transcript: str) -> dict:
    """Generate additional content"""
    res = client.beta.chat.completions.parse(
        model = "gpt-4.1",
        messages = [
            {
                "role": "system", 
                "content": 
                    """
                    You are a course creator for a college class. 
                    Generate a headline, summary, todo list, glossary, and quiz that reflects the following transcript.
                    Indicate subheaders using '##', bold key terms and concepts for emphasis, use bullet points to represent lists, and indentation to indicate dependencies.
                    Format the response in markdown. Format math equations and symbols in katex.
                    """
            },
            {
                "role": "user", 
                "content": transcript
            }
        ],
        response_format = Extras
    )
    extras = res.choices[0].message.parsed.model_dump()
    return extras

def hallucination_check(note_section_text: str, transcript_section_text: str) -> dict:
    """Evaluate the accuracy of a note section against its corresponding transcript section"""
    res = client.beta.chat.completions.parse(
        model = "gpt-4.1",
        messages = [
            {
                "role": "system", 
                "content": 
                    """
                    You are a course creator for a college class. 
                    Evaluate the accuracy of an AI generated note section against the transcript section it represents.
                    """
            },
            {
                "role": "user", 
                "content": f"note section: {note_section_text} \n transcript section: {transcript_section_text}"
            }
        ],
        response_format = Eval
    )
    evaluation = res.choices[0].message.parsed.model_dump()
    return evaluation

def fix_hallucination(note_section_text: str, transcript_section_text: str, hallucinations: list[str], revisions: list[str]) -> str:
    """Make revisions to resolve hallucinations"""
    res = client.beta.chat.completions.parse(
        model = "gpt-4.1",
        messages = [
            {
                "role": "system", 
                "content": 
                    """
                    You are a course creator for a college class. 
                    Given the following set of hallucinations and revisions that should be made, revise the AI generated note section such that it accurately represents the transcript section.
                    Maintain the style and formatting of the note section, and avoid making sweeping changes.
                    """
            },
            {
                "role": "user", 
                "content": 
                    f"""
                    note section: {note_section_text} 
                    transcript section: {transcript_section_text} 
                    hallucinations: {str(hallucinations)}
                    revisions: {str(revisions)}
                    """
            }
        ]
    )
    revised = res.choices[0].message.content
    return revised

def self_reflection(note_section_text: str, transcript_section_text: str, max_iterations: int, threshold: int):
    """Iteratively evaluate and revise note until a threshold or plateau is reached"""
    iteration = 0
    net_gain = 0
    scores = []
    note_sections = []
    max_index = 0

    while iteration < max_iterations:
        if scores and scores[-1] >= threshold:
            break
        if len(scores) >= 2:
            net_gain += scores[-1] - scores[-2]
        if net_gain <= -10:
            break

        eval = hallucination_check(note_section_text, transcript_section_text)
        scores.append(eval["score"])

        if scores[iteration] > scores[max_index]:
            max_index = iteration

        note_sections.append(note_section_text)
        note_section_text = fix_hallucination(note_section_text, 
                                              transcript_section_text, 
                                              eval["hallucinations"], 
                                              eval["revisions"]
                                              )
        iteration += 1

    return note_sections[max_index]

def compile_note(transcript: str) -> str:
    """
    Break a transcript into semantically meaningful sections
    Generate a note for each section, and additional content for the transcript
    Combine each note and additional content together
    """
    split_transcript = split_transcript_into_lines(transcript)
    messages = [
        {
            "role": "system", 
            "content": "Break the following transcript into 5 or 6 sections using the provided line numbers."
        },
        {
            "role": "user", 
            "content": str(split_transcript)
        }
    ]
    transcript_sections = annotate_transcript(split_transcript, messages)
    compiled_note_sections = []

    for transcript_section in transcript_sections:
        normalized_section = normalize_section(transcript_section, split_transcript)
        note_section = generate_note_section(normalized_section)
        compiled_note = self_reflection(compile_note_section(note_section), normalized_section["transcript"], max_iterations = 4, threshold = 0.98)
        compiled_note_sections.append(compiled_note)

    extras = generate_extras(transcript)

    combined_note_sections = ""
    for compiled_note_section in compiled_note_sections:
        combined_note_sections += compiled_note_section + "\n"

    combined_note = (
                        extras["headline"] + 
                        "\n" +
                        extras["summary"] + 
                        "\n" +
                        extras["todo_list"] + 
                        "\n" +
                        combined_note_sections +
                        extras["glossary"] +
                        "\n" +
                        extras["quiz"] +
                        "\n" +
                        extras["answers"]
                    )
    return combined_note