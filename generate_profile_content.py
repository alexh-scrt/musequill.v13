#!/usr/bin/env python3
"""
CLI for profile and depth-aware content generation.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.profile_aware_generator import ProfileAwareGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(
        description='Generate depth-differentiated content with profile awareness'
    )
    
    parser.add_argument(
        'topic',
        type=str,
        help='Topic to generate content about'
    )
    
    parser.add_argument(
        '--profile',
        type=str,
        choices=['scientific', 'popular_science', 'educational'],
        default='scientific',
        help='Content profile (default: scientific)'
    )
    
    parser.add_argument(
        '--depth',
        type=int,
        choices=[1, 2, 3],
        default=3,
        help='Maximum depth level to generate (default: 3)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (optional)'
    )
    
    parser.add_argument(
        '--single-depth',
        type=int,
        choices=[1, 2, 3],
        help='Generate only a single depth level (optional)'
    )
    
    parser.add_argument(
        '--session-id',
        type=str,
        default=None,
        help='Session ID for tracking (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create session ID if not provided
    session_id = args.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize generator
    print(f"Initializing profile-aware generator...")
    generator = ProfileAwareGenerator(session_id=session_id)
    
    # Generate content
    print(f"Generating content about '{args.topic}'")
    print(f"Profile: {args.profile}, Max Depth: {args.depth}")
    print("-" * 50)
    
    try:
        if args.single_depth:
            # Generate only a single depth
            print(f"Generating only depth {args.single_depth}...")
            content, metadata = await generator.generate_depth_section(
                topic=args.topic,
                profile=args.profile,
                depth=args.single_depth,
                previous_depths=None  # No context for single depth
            )
            
            # Output result
            print(f"\n### Depth {args.single_depth} Content ###\n")
            print(content)
            print(f"\n### Metadata ###")
            print(f"Decision: {metadata.get('decision', 'UNKNOWN')}")
            print(f"Attempts: {metadata.get('attempts', 0)}")
            if 'similarity_score' in metadata:
                print(f"Similarity Score: {metadata['similarity_score']:.2%}")
            
            # Save if output path provided
            if args.output:
                output_path = Path(args.output)
                output_path.write_text(content)
                print(f"\nContent saved to {output_path}")
        
        else:
            # Generate full document
            print(f"Generating depths 1 to {args.depth}...")
            document = await generator.generate_document(
                topic=args.topic,
                profile=args.profile,
                max_depth=args.depth
            )
            
            # Output result
            if args.output:
                output_path = Path(args.output)
            else:
                # Generate default output filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = Path(f"outputs/profile_content_{args.profile}_{timestamp}.md")
                output_path.parent.mkdir(exist_ok=True)
            
            # Save document
            document.save(str(output_path))
            print(f"\nDocument saved to {output_path}")
            
            # Display summary
            print("\n### Generation Summary ###")
            for depth in sorted(document.metadata.keys()):
                meta = document.metadata[depth]
                print(f"Depth {depth}:")
                print(f"  - Decision: {meta.get('decision', 'UNKNOWN')}")
                print(f"  - Attempts: {meta.get('attempts', 0)}")
                if 'similarity_score' in meta:
                    print(f"  - Similarity: {meta['similarity_score']:.2%}")
                if meta.get('flagged'):
                    print(f"  - Warning: {meta.get('reason', 'Flagged for review')}")
            
            # Show a preview
            print("\n### Document Preview (first 500 chars) ###")
            preview = document.render()[:500]
            print(preview)
            if len(document.render()) > 500:
                print("...")
    
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    # Run async main
    exit_code = asyncio.run(main())
    sys.exit(exit_code)