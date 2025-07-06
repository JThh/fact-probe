#!/usr/bin/env python3
"""
FactScore Data Processor

A utility for processing FactScore evaluation results and converting them into
a structured Python data format. This tool extracts claims and their labels
from FactScore pickle files and generates a Python file with the processed data.

Usage:
    python factscore2data.py --factscore-file path/to/factscore.pkl --entities-file path/to/entities.txt --output output.py

Author: Your Name
License: MIT
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


# Default configuration
DEFAULT_CONFIG = {
    'major_false_label': 'Major False',
    'minor_false_label': 'Minor False',
    'flag_words': ['speaker', 'appear', 'incomplete', 'provide', 'want', 'bio', 'text'],
    'question_template': 'Who is {entity}? Provide as many specific details and examples as possible (such as names of people, numbers, events, locations, dates, times, etc.)'
}


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_pickle_file(file_path: Path) -> Dict[str, Any]:
    """
    Load a pickle file and return its content.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        The loaded data from the pickle file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pickle.UnpicklingError: If the file cannot be unpickled
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f'Successfully loaded data from "{file_path}"')
        return data
    except FileNotFoundError:
        logging.error(f'File "{file_path}" not found')
        raise
    except pickle.UnpicklingError:
        logging.error(f'Failed to unpickle file "{file_path}"')
        raise


def load_entities_file(file_path: Path) -> List[str]:
    """
    Load entities from a text file (one per line).
    
    Args:
        file_path: Path to the entities file
        
    Returns:
        List of entity names
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            entities = [line.strip() for line in f if line.strip()]
        logging.info(f'Loaded {len(entities)} entities from "{file_path}"')
        return entities
    except FileNotFoundError:
        logging.error(f'File "{file_path}" not found')
        raise


def is_valid_claim(claim: str, flag_words: List[str]) -> bool:
    """
    Check if a claim is valid by ensuring it doesn't contain flag words.
    
    Args:
        claim: The claim text to validate
        flag_words: List of words that invalidate a claim
        
    Returns:
        True if the claim is valid, False otherwise
    """
    if not claim or not claim.strip():
        return False
    
    claim_lower = claim.lower()
    return not any(word in claim_lower for word in flag_words)


def process_entity_claims(
    entities: List[str],
    decisions: List[List[Dict[str, Any]]],
    config: Dict[str, Any]
) -> Tuple[List[List[Any]], int]:
    """
    Process claims for each entity and compile the structured data.
    
    Args:
        entities: List of entity names
        decisions: FactScore decisions for each entity
        config: Configuration dictionary
        
    Returns:
        Tuple of (processed_data, total_claims_count)
    """
    data = []
    total_claims = 0
    
    question_template = config['question_template']
    major_false_label = config['major_false_label']
    flag_words = config['flag_words']

    for i, (entity, claims) in enumerate(zip(entities[:len(decisions)], decisions)):
        if not claims:
            logging.warning(f'No claims found for entity "{entity}" (index {i})')
            continue
            
        # Initialize datum with basic structure
        datum = [
            i,
            question_template.format(entity=entity),
            None,
            [None]
        ]

        # Extract valid claims with their support labels
        valid_claims_with_labels = []
        for claim in claims:
            claim_text = claim.get('atom', '').strip()
            is_supported = claim.get('is_supported', False)
            
            if is_valid_claim(claim_text, flag_words):
                valid_claims_with_labels.append((claim_text, is_supported))

        if not valid_claims_with_labels:
            logging.warning(f'No valid claims found for entity "{entity}" (index {i})')
            # Append empty lists for atoms and labels
            datum.extend([[], []])
            data.append(datum)
            continue

        # Separate claims and labels
        claims_text = [claim[0] for claim in valid_claims_with_labels]
        claims_labels = [
            major_false_label if not claim[1] else claim[1] 
            for claim in valid_claims_with_labels
        ]

        datum.extend([claims_text, claims_labels])
        data.append(datum)
        total_claims += len(claims_text)
        
        logging.debug(f'Processed entity "{entity}": {len(claims_text)} valid claims')

    return data, total_claims


def write_python_output(
    data: List[List[Any]],
    output_path: Path,
    config: Dict[str, Any]
) -> None:
    """
    Write the processed data to a Python file.
    
    Args:
        data: Processed data for each entity
        output_path: Path to the output Python file
        config: Configuration dictionary
        
    Raises:
        IOError: If the file cannot be written
    """
    major_false_label = config['major_false_label']
    minor_false_label = config['minor_false_label']
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header comment
            f.write('"""\n')
            f.write('Generated FactScore data file\n')
            f.write('This file contains processed claims and their labels from FactScore evaluation.\n')
            f.write('"""\n\n')
            
            # Write constants
            f.write(f"MAJOR = '{major_false_label}'\n")
            f.write(f"MINOR = '{minor_false_label}'\n\n")
            
            # Write data structure documentation
            f.write('# Data structure: [index, question, None, [None], claims_list, labels_list]\n')
            f.write('data = [\n')
            
            for datum in data:
                f.write('    [\n')
                # Write basic fields
                f.write(f'        {datum[0]},\n')
                f.write(f'        """{datum[1]}""",\n')
                f.write('        None,\n')
                f.write('        [None],\n')
                
                # Write claims list
                f.write('        [\n')
                for claim in datum[4]:
                    escaped_claim = claim.replace("'", "\\'").replace('\n', '\\n')
                    f.write(f"            '{escaped_claim}',\n")
                f.write('        ],\n')
                
                # Write labels list
                f.write('        [\n')
                for label in datum[5]:
                    if label == major_false_label:
                        f.write('            MAJOR,\n')
                    elif label == minor_false_label:
                        f.write('            MINOR,\n')
                    else:
                        f.write(f'            {repr(label)},\n')
                f.write('        ]\n')
                f.write('    ],\n')
                
            f.write(']\n')
            
        logging.info(f'Successfully wrote processed data to "{output_path}"')
        
    except IOError as e:
        logging.error(f'Failed to write output file "{output_path}": {e}')
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process FactScore evaluation results into structured Python data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --factscore-file results.pkl --entities-file entities.txt --output data.py
  %(prog)s -f results.pkl -e entities.txt -o data.py --verbose
        """
    )
    
    parser.add_argument(
        '-f', '--factscore-file',
        type=Path,
        required=True,
        help='Path to the FactScore pickle file containing evaluation results'
    )
    
    parser.add_argument(
        '-e', '--entities-file',
        type=Path,
        required=True,
        help='Path to the text file containing entity names (one per line)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='Path for the output Python file'
    )
    
    parser.add_argument(
        '--major-label',
        default=DEFAULT_CONFIG['major_false_label'],
        help=f'Label for major false claims (default: {DEFAULT_CONFIG["major_false_label"]})'
    )
    
    parser.add_argument(
        '--minor-label',
        default=DEFAULT_CONFIG['minor_false_label'],
        help=f'Label for minor false claims (default: {DEFAULT_CONFIG["minor_false_label"]})'
    )
    
    parser.add_argument(
        '--question-template',
        default=DEFAULT_CONFIG['question_template'],
        help='Template for generating questions (use {entity} as placeholder)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to orchestrate the data processing."""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    # Create configuration from arguments and defaults
    config = DEFAULT_CONFIG.copy()
    config.update({
        'major_false_label': args.major_label,
        'minor_false_label': args.minor_label,
        'question_template': args.question_template,
    })
    
    try:
        # Load input data
        logging.info('Loading FactScore data...')
        factscore_data = load_pickle_file(args.factscore_file)
        
        logging.info('Loading entities...')
        entities = load_entities_file(args.entities_file)
        
        # Validate FactScore data structure
        decisions = factscore_data.get('decisions', [])
        if not decisions:
            logging.error('No "decisions" key found in FactScore data')
            sys.exit(1)
        
        if len(entities) < len(decisions):
            logging.warning(f'More decisions ({len(decisions)}) than entities ({len(entities)})')
        elif len(entities) > len(decisions):
            logging.warning(f'More entities ({len(entities)}) than decisions ({len(decisions)})')
        
        # Process the data
        logging.info('Processing claims...')
        processed_data, total_claims = process_entity_claims(entities, decisions, config)
        
        # Write output
        logging.info(f'Writing {len(processed_data)} processed entities with {total_claims} total claims...')
        write_python_output(processed_data, args.output, config)
        
        logging.info('Processing completed successfully!')
        logging.info(f'Summary: {len(processed_data)} entities, {total_claims} claims')
        
    except Exception as e:
        logging.error(f'Processing failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
