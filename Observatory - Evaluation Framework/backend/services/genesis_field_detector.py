"""
Genesis Feasibility Document Detector

Identifies whether uploaded documents are Genesis Capital feasibility review files
based on filename patterns, content markers, and document types.
"""
import os
import re
import logging

logger = logging.getLogger(__name__)

# Genesis-specific filename patterns that indicate feasibility documents
GENESIS_FILENAME_PATTERNS = [
    r"G\d{8}",  # Genesis deal ID format (e.g., G25069857)
    r"S\d{4}",  # Sponsor ID format (e.g., S4855)
    r"Construction.*Budget",
    r"Construction.*Timeline",
    r"Feasibility.*Report",
    r"Trinity.*Inspection",
    r"Sponsor.*Construction.*Analysis",
    r"SCA",
    r"Copy.*of.*Plans",
    r"deal.*notes",
]

# Genesis-specific content markers that appear in feasibility documents
GENESIS_CONTENT_MARKERS = [
    "Genesis Capital",
    "Trinity Inspection Services",
    "Feasibility Review",
    "Construction Holdback",
    "Project Address / Title",
    "Borrower Entity",
    "Sponsor Construction Analysis",
    "Third-Party Review",
    "Gross Buildable Square Footage",
    "Property Type",
    "Rehab Amount",
]

# Expected document types in a Genesis feasibility package
GENESIS_DOCUMENT_TYPES = {
    "trinity": ["feasibility", "report", "inspection"],
    "budget": ["budget", ".xls", ".xlsx"],
    "timeline": ["timeline", "schedule", "gantt"],
    "plans": ["plans", "drawings", ".pdf"],
    "sca": ["sponsor", "construction", "analysis"],
}


def _matches_genesis_filename(filename: str) -> bool:
    """Check if filename matches Genesis feasibility document patterns."""
    filename_lower = filename.lower()
    for pattern in GENESIS_FILENAME_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return True
    return False


def _contains_genesis_markers(text: str, threshold: int = 2) -> bool:
    """Check if document content contains Genesis-specific markers.
    
    Args:
        text: Document text content
        threshold: Minimum number of markers required (default 2)
    
    Returns:
        True if threshold number of markers found
    """
    if not text:
        return False
    
    marker_count = 0
    text_lower = text.lower()
    
    for marker in GENESIS_CONTENT_MARKERS:
        if marker.lower() in text_lower:
            marker_count += 1
            if marker_count >= threshold:
                return True
    
    return False


def _categorize_genesis_document(filename: str) -> str | None:
    """Categorize a Genesis document by type (trinity, budget, timeline, etc.)."""
    filename_lower = filename.lower()
    
    for doc_type, keywords in GENESIS_DOCUMENT_TYPES.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return doc_type
    
    return None


def is_genesis_feasibility_package(document_paths: list[str]) -> tuple[bool, dict]:
    """
    Detect if uploaded documents constitute a Genesis feasibility package.
    
    NOTE: "Copy of Plans" files are included in detection but excluded from Q&A generation
    via document_reader.document_paths() filter.
    
    Args:
        document_paths: List of document file paths
    
    Returns:
        Tuple of (is_genesis, details_dict) where details_dict contains:
        - is_genesis: Boolean indicating if this is a Genesis package
        - confidence: Float between 0-1 indicating detection confidence
        - detected_types: List of detected document types
        - total_files: Total number of files analyzed
        - genesis_files: Number of files identified as Genesis documents
    """
    from backend.services.document_reader import read_document, display_name
    
    if not document_paths:
        return False, {"confidence": 0.0, "reason": "No documents provided"}
    
    genesis_file_count = 0
    detected_types = set()
    content_checks_passed = 0
    total_files = len(document_paths)
    
    for path in document_paths:
        if not os.path.exists(path):
            continue
            
        filename = display_name(path)
        
        # Check filename patterns
        if _matches_genesis_filename(filename):
            genesis_file_count += 1
            doc_type = _categorize_genesis_document(filename)
            if doc_type:
                detected_types.add(doc_type)
        
        # Check content for markers (only for text-based files, avoid large PDFs)
        ext = os.path.splitext(path)[1].lower()
        if ext in [".txt", ".docx", ".doc"] and os.path.getsize(path) < 5_000_000:  # 5MB limit
            try:
                text = read_document(path)
                if _contains_genesis_markers(text, threshold=2):
                    content_checks_passed += 1
                    genesis_file_count += 1
            except Exception as e:
                logger.debug(f"Could not read {filename} for content check: {e}")
    
    # Calculate confidence based on multiple signals
    filename_ratio = genesis_file_count / total_files if total_files > 0 else 0
    has_key_documents = bool(detected_types & {"trinity", "budget", "sca"})
    
    # Decision logic
    is_genesis = False
    confidence = 0.0
    reasons = []
    
    if filename_ratio >= 0.5:  # At least 50% of files match Genesis patterns
        is_genesis = True
        confidence = min(0.95, 0.5 + (filename_ratio * 0.3) + (0.15 if has_key_documents else 0))
        reasons.append(f"{genesis_file_count}/{total_files} files match Genesis patterns")
    
    if content_checks_passed >= 1:
        is_genesis = True
        confidence = max(confidence, 0.7)
        reasons.append(f"{content_checks_passed} file(s) contain Genesis feasibility markers")
    
    if has_key_documents:
        reasons.append(f"Key document types detected: {', '.join(detected_types)}")
    
    details = {
        "is_genesis": is_genesis,
        "confidence": round(confidence, 2),
        "total_files": total_files,
        "genesis_files": genesis_file_count,
        "detected_types": list(detected_types),
        "reasons": reasons,
    }
    
    if is_genesis:
        logger.info(f"Genesis feasibility package detected: {details}")
    else:
        logger.debug(f"Not a Genesis feasibility package: {details}")
    
    return is_genesis, details


# The 39 Genesis feasibility fields for focused Q&A generation
GENESIS_39_FIELDS = """
## 1. Report Header (6 Fields)
1. Date of Report Approved
2. Project Address / Title
3. Sponsor
4. Borrower Entity
5. Project Status
6. Report Created by

## 2. Executive Summary (12 Fields)
7. Additional Comments
8. Third-Party Review
9. Third-Party Reviewer
10. Third-party Review (Good/Bad)
11. Third-Party Review (Meet or Fail)
12. Genesis Agree (Y/N)
13. Project Timeline To Date
14. Remaining Timeline
15. appropriate / not appropriate (Timeline)
16. Draw Hold
17. Specify Draw Hold Items
18. Other Special Conditions

## 3. Loan Summary (15 Fields)
19. Project Type
20. Rehab Amount
21. Construction Holdback Amount
22. Project Cost per Square Foot
23. Cost per Structure
24. Cost per Unit
25. Contingency Amount
26. Contingency (%)
27. Project Complete Percentage
28. Budget Review
29. Additional Budget Comments
30. Plan Status
31. Plan Review Status
32. Permit Status
33. Permits (Post Funding)

## 4. Finished Product Details (6 Fields)
34. Property Type
35. Region
36. No. of Units
37. No. of Stories
38. No. of Structures
39. Gross Buildable Square Footage (GFA)
"""
