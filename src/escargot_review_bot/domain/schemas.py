from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser


class ReviewRequest(BaseModel):
    base_sha: str
    head_sha: str
    pull_request_number: int


class LineMapping(BaseModel):
    target_id: int
    line_type: Literal['added', 'removed', 'context']
    content: str
    source_line_no: Optional[int] = None
    target_line_no: Optional[int] = None


class LLMReviewComment(BaseModel):
    """A single review comment from the LLM."""
    target_id: int = Field(description="The line ID from the Commentable Catalog")
    body: str = Field(description="The review comment content (3-8 sentences)")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)


class JudgeComment(BaseModel):
    """A merged comment from the Judge pass."""
    body: str = Field(description="The integrated review comment content")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)


class GitHubComment(BaseModel):
    path: str
    body: str
    commit_id: str
    line: int
    side: Literal["LEFT", "RIGHT"]


review_comment_parser = JsonOutputParser(pydantic_object=LLMReviewComment)
judge_comment_parser = JsonOutputParser(pydantic_object=JudgeComment)